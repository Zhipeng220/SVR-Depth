#!/usr/bin/env python3
"""Audit existing M01 frame and temporal CSVs without rerunning inference."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    source = a.result_dir.resolve(strict=True)
    frame_path, temporal_path = source / 'frame_metrics.csv', source / 'temporal_metrics.csv'
    with frame_path.open(encoding='utf-8-sig', newline='') as f: frames = list(csv.DictReader(f))
    with temporal_path.open(encoding='utf-8-sig', newline='') as f: pairs = list(csv.DictReader(f))
    grouped, keys = collections.defaultdict(list), {}
    duplicates = []
    for row in frames:
        group = (row['dataset'], row['method'])
        grouped[group].append(row)
        key = (row['dataset'], row['sequence'], row['index'], row['method'])
        if key in keys: duplicates.append(key)
        keys[key] = row
    if duplicates: raise ValueError(f'Duplicate method/frame identities: {len(duplicates)}')
    summaries = []
    for (dataset, method), rows in sorted(grouped.items()):
        def mean(field):
            vals = []
            for row in rows:
                try: value = float(row[field])
                except (ValueError, KeyError): continue
                if math.isfinite(value): vals.append(value)
            return {'mean': statistics.fmean(vals) if vals else None, 'valid_frames': len(vals)}
        selected = [x for x in pairs if x['dataset'] == dataset and x['method'] == method]
        summaries.append({'dataset': dataset, 'method': method, 'frames': len(rows),
            'sequences': len({r['sequence'] for r in rows}),
            'frame_status': dict(collections.Counter(r['status'] for r in rows)),
            'raw_delta1': mean('delta1'), 'upstream_delta1_ssi': mean('upstream_depth_d1_ssi'),
            'absrel': mean('absrel'), 'rmse': mean('rmse'), 'temporal_records': len(selected),
            'temporal_status': dict(collections.Counter(r['status'] for r in selected)),
            'valid_provisional_tau_pairs': sum(r.get('status') == 'ok' and bool(r.get('tau5_provisional')) for r in selected)})
    arows = {(r['dataset'], r['sequence'], r['index']): r for r in frames if r['method'] == 'A'}
    prows = {(r['dataset'], r['sequence'], r['index']): r for r in frames if r['method'] == 'P_multi'}
    shared = set(arows) & set(prows)
    mismatches = sum(any(arows[k].get(f) != prows[k].get(f) for f in ('rgb_sha256', 'gt_sha256', 'mask_sha256')) for k in shared)
    seqs = collections.defaultdict(list)
    for row in arows.values(): seqs[(row['dataset'], row['sequence'])].append(row)
    manifest = []
    for (dataset, sequence), rows in sorted(seqs.items()):
        rows.sort(key=lambda r: int(r['index']))
        indices = [int(r['index']) for r in rows]
        if len(indices) != len(set(indices)): raise ValueError('Duplicate indices')
        manifest.append({'dataset': dataset, 'sequence': sequence,
            'scene': rows[0]['scene'], 'full_sequence_count': len(rows),
            'full_sequence': [{'source_frame_index': r['source_frame_index'], 'source_frame_id': r['source_frame_id']} for r in rows],
            'release_last32': [{'source_frame_index': r['source_frame_index'], 'source_frame_id': r['source_frame_id']} for r in rows[-32:]],
            'development_overlap': dataset == 'Sintel' and sequence.split('/')[-1] == 'shaman_3'})
    result = {'scope': 'historical_full_sequence_k16_diagnostic_not_paper',
              'frame_csv_sha256': sha(frame_path), 'temporal_csv_sha256': sha(temporal_path),
              'a_frames': len(arows), 'p_frames': len(prows), 'paired_frames': len(shared),
              'unpaired_a': len(set(arows) - set(prows)), 'unpaired_p': len(set(prows) - set(arows)),
              'paired_rgb_gt_mask_mismatches': mismatches, 'datasets': summaries,
              'limitations': ['provisional temporal values are not author verified',
                              'full-sequence custom preprocessing differs from upstream release protocol']}
    out = a.out_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out / 'existing_results_review.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    (out / 'frame_manifest.json').write_text(json.dumps({'source_frame_csv_sha256': sha(frame_path),
        'scope': 'derived_from_existing_full_sequence_run_not_a_paper_split', 'sequences': manifest}, indent=2), encoding='utf-8')
    lines = ['# M01 existing full-sequence K16 review', '',
             'Diagnostic only; not an author-protocol paper result.', '',
             f"Paired frames: {len(shared)}; unpaired A: {len(set(arows)-set(prows))}; unpaired P: {len(set(prows)-set(arows))}; RGB/GT/mask mismatches: {mismatches}.", '',
             '| Dataset | Method | Frames | Sequences | raw δ1 | δ1 SSI | temporal rows | valid provisional τ5 pairs |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        fmt = lambda x: 'unavailable' if x is None else f'{100*x:.3f}%'
        lines.append(f"| {r['dataset']} | {r['method']} | {r['frames']} | {r['sequences']} | {fmt(r['raw_delta1']['mean'])} | {fmt(r['upstream_delta1_ssi']['mean'])} | {r['temporal_records']} | {r['valid_provisional_tau_pairs']} |")
    lines += ['', 'SSI and preprocessing are not yet proven equivalent to the paper; temporal rows do not imply valid temporal scores.',
              'The last-32 manifest is derived from this run and does not independently establish the cited paper split.', '']
    (out / 'M01_existing_results_review.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out / 'M01_existing_results_review.md')


if __name__ == '__main__': main()
