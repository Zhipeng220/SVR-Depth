#!/usr/bin/env python3
"""Offline, one-intervention O/T oracle diagnostic from M05 artifacts.

Future GT is used: this is neither an online policy nor a full-trajectory bound.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

KEY = ('seed', 'sequence', 'cycle', 'age', 'parent', 'anchor_index')
BRANCHES = ('A', 'O', 'F', 'T')


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def index(rows, extra):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in KEY + extra)
        if key in out:
            raise ValueError(f'duplicate row: {key}')
        out[key] = row
    return out


def equal_value(a, b):
    if a == b:
        return True
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except (ValueError, TypeError):
        return False


def analyze(future, state, current, summary, deadband=0.001):
    if not math.isfinite(deadband) or deadband < 0:
        raise ValueError('deadband must be finite and nonnegative')
    if summary.get('all_invariants_pass') is not True or summary.get('failed_invariants'):
        raise ValueError('source M05 invariants not confirmed')
    if summary.get('K') != 16 or summary.get('H') != 4:
        raise ValueError('expected K16, H4 M05 source')
    fi, si, ci = index(future, ('branch', 'horizon')), index(state, ('branch',)), index(current, ('branch',))
    anchors = sorted({k[:len(KEY)] for k in si if k[-1] in BRANCHES})
    if not anchors:
        raise ValueError('no anchors')
    if summary.get('invariant_checks') != len(anchors):
        raise ValueError('anchor count differs from invariant audit')
    records = []
    for key in anchors:
        if int(key[3]) + 4 >= 16:
            raise ValueError(f'anchor horizon crosses refresh: {key}')
        try:
            states = {b: si[key + (b,)] for b in BRANCHES}
            cur = {b: ci[key + (b,)] for b in ('O', 'T')}
        except KeyError as e:
            raise ValueError(f'missing state or current branch: {key}') from e
        hashes = [states[b]['parent_state_sha256'] for b in BRANCHES]
        if not hashes[0] or len(set(hashes)) != 1:
            raise ValueError(f'parent mismatch: {key}')
        if states['A']['committed_state_sha256'] != states['O']['committed_state_sha256'] or states['F']['committed_state_sha256'] != states['T']['committed_state_sha256']:
            raise ValueError(f'committed state invariant failure: {key}')
        for b in ('O', 'T'):
            if cur[b]['committed_state_sha256'] != states[b]['committed_state_sha256'] or cur[b]['current_output_source'] != 'D1':
                raise ValueError(f'current output/state mismatch: {key} {b}')
        for col in ('gt_matched', 'AbsRel', 'RMSE', 'delta1', 'valid_pixels', 'total_pixels'):
            if not equal_value(cur['O'][col], cur['T'][col]):
                raise ValueError(f'current D1 metric mismatch: {key} {col}')
        for h in range(1, 5):
            o, t = fi.get(key + ('O', str(h))), fi.get(key + ('T', str(h)))
            record = dict(zip(KEY, key))
            record.update(horizon=h, valid=False, reason='', absrel_O=None, absrel_T=None, delta_T_minus_O=None, local_oracle=None, choice=None)
            if o is None or t is None:
                record['reason'] = 'missing_branch_or_horizon'
                records.append(record)
                continue
            if o['frame_index'] != t['frame_index'] or int(o['frame_index']) != int(key[-1]) + h:
                raise ValueError(f'future frame mismatch: {key} h={h}')
            for row in (o, t):
                if row['repair_mode_after_intervention'] != 'off' or row['static_cache_unchanged'].lower() != 'true':
                    raise ValueError(f'future replay contract failure: {key} h={h}')
            if o['gt_matched'].lower() != 'true' or t['gt_matched'].lower() != 'true':
                record['reason'] = 'unmatched_gt'
                records.append(record)
                continue
            if o['valid_pixels'] != t['valid_pixels'] or int(o['valid_pixels']) <= 0 or o['total_pixels'] != t['total_pixels']:
                raise ValueError(f'pixel-count mismatch: {key} h={h}')
            try:
                a, b = float(o['AbsRel']), float(t['AbsRel'])
            except ValueError:
                record['reason'] = 'missing_absrel'
                records.append(record)
                continue
            if not all(math.isfinite(x) and x >= 0 for x in (a, b)):
                record['reason'] = 'invalid_absrel'
                records.append(record)
                continue
            d = b-a
            record.update(valid=True, absrel_O=a, absrel_T=b, delta_T_minus_O=d,
                          local_oracle=min(a,b), choice='O' if d > deadband else ('T' if d < -deadband else 'near_tie'))
            records.append(record)
    groups = defaultdict(list)
    for row in records:
        groups[(row['seed'], row['parent'], row['sequence'], row['horizon'])].append(row)
    grouped = []
    for (seed, parent, seq, h), vals in sorted(groups.items()):
        valid = [r for r in vals if r['valid']]
        row = dict(seed=seed, parent=parent, sequence=seq, horizon=h, total=len(vals), valid=len(valid),
                   fraction_valid=len(valid)/len(vals),
                   n_O=sum(r['choice']=='O' for r in valid), n_T=sum(r['choice']=='T' for r in valid),
                   n_near_tie=sum(r['choice']=='near_tie' for r in valid))
        for field in ('absrel_O','absrel_T','delta_T_minus_O','local_oracle'):
            row['mean_'+field] = mean(r[field] for r in valid) if valid else None
        grouped.append(row)
    macro = []
    keys = sorted({(r['seed'], r['parent'], r['horizon']) for r in grouped})
    for seed, parent, h in keys:
        vals = [r for r in grouped if (r['seed'],r['parent'],r['horizon']) == (seed,parent,h) and r['valid']]
        m = dict(seed=seed,parent=parent,horizon=h,sequences=len(vals))
        for field in ('absrel_O','absrel_T','delta_T_minus_O','local_oracle'):
            m['mean_'+field] = mean(r['mean_'+field] for r in vals) if vals else None
        macro.append(m)
    return records, grouped, macro


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--deadband', type=float, default=0.001)
    args=parser.parse_args()
    inp=args.input_dir.resolve(); out=args.out_dir.resolve()
    if out == inp or inp in out.parents:
        raise ValueError('output must be outside input directory')
    paths={name:inp/name for name in ('m05_future_frame_metrics.csv','m05_state_audit.csv','m05_current_frame_metrics.csv','m05_development_summary.json')}
    summary=json.loads(paths['m05_development_summary.json'].read_text(encoding='utf-8-sig'))
    records, grouped, macro=analyze(read_rows(paths['m05_future_frame_metrics.csv']), read_rows(paths['m05_state_audit.csv']), read_rows(paths['m05_current_frame_metrics.csv']), summary, args.deadband)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out/'paired_horizons.csv',records)
    write_csv(out/'sequence_summary.csv',grouped)
    anchor_groups = defaultdict(list)
    for row in records:
        anchor_groups[tuple(row[k] for k in KEY)].append(row)
    windows = []
    for key, vals in sorted(anchor_groups.items()):
        valid = [r for r in vals if r['valid']]
        row = dict(zip(KEY, key))
        row.update(valid_horizons=len(valid), complete_h1_h4=len(valid)==4,
                   mean_delta_T_minus_O=mean(r['delta_T_minus_O'] for r in valid) if valid else None)
        windows.append(row)
    write_csv(out/'anchor_window_summary.csv', windows)
    result={'scope':'offline single intervention local oracle; future GT privileged; not full trajectory or online policy',
            'development_only':True, 'deadband_absrel':args.deadband, 'input_sha256':{k:digest(p) for k,p in paths.items()},
            'analysis_script_sha256':digest(Path(__file__).resolve()),
            'contract_limit':'Equal parent/committed hashes, D1 metrics and source invariant summary checked; individual tensor identity and pixel masks cannot be independently reconstructed from these CSVs.',
            'cost_limit':'O and T both require repair candidate D1; this diagnostic establishes no runtime savings.',
            'macro_equal_sequence':macro, 'n_anchor_parents':len(records)//4}
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (out/'REPORT.md').write_text('# M05 local oracle diagnostic\n\nOne-step offline intervention, using future GT; not a complete trajectory oracle or deployable policy.\n\nO and T both compute D1; no speedup is established. Source invariants were checked but individual tensor values and pixel masks cannot be reconstructed from CSV. See summary.json and sequence_summary.csv.\n',encoding='utf-8')
    print(f'Wrote {out}; anchor-parent records={len(records)//4}')


if __name__ == '__main__':
    main()
