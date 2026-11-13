"""Y05-R frozen protocol and stdlib helpers; imports no Torch at module import."""
from pathlib import Path
import csv, hashlib, json, math, os, sys

VERSION = 'Y05-R-1.0.1'
ARMS = ('C_star', 'CF')
SEQS = ('crowd2', 'crowd3', 'balloon2')
N_RGB = {'crowd2': 895, 'crowd3': 854, 'balloon2': 469}
TAIL = {'crowd2': 768, 'crowd3': 256, 'balloon2': 256}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def jread(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jwrite(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    t.replace(p)


def cread(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def cwrite(path, rows):
    require(bool(rows), 'Refusing to write empty table: ' + str(path))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with t.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)
    t.replace(p)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def truth(value):
    return str(value).strip().lower() in ('true', '1', 'yes')


def matched(row):
    return str(row.get('gt_match_status', '')).strip().upper() == 'MATCHED'


def source_y05_dir(explicit, root, identity):
    if explicit != 'auto':
        choices = [Path(explicit).resolve()]
    else:
        choices = list((Path(root) / 'outputs').glob('gt4a2_y05_*/validation'))
    found = []
    for p in choices:
        if not ((p / 'summary.json').is_file() and (p / 'protocol.json').is_file()):
            continue
        try:
            summary = jread(p / 'summary.json')
            if (summary.get('status') == 'Y05_DEVELOPMENT_SUITE_COMPLETE'
                    and not summary.get('partial_smoke')
                    and sha(p / 'protocol.json') == identity['source_protocol_sha256']
                    and sha(p / 'summary.json') == identity['source_summary_sha256']
                    and sha(p / 'y05_frame_metrics.csv') == identity['source_frame_metrics_sha256']):
                found.append(p.resolve())
        except Exception:
            continue
    require(bool(found), 'Reviewed complete Y05 run not found/changed. Set Y05_DIR to the original Y05 validation directory; do not bypass identity checks.')
    return max(found, key=lambda p: (p / 'summary.json').stat().st_mtime_ns)


def frozen_settings(smoke=False):
    return dict(
        horizon=4, train_K=32, ages=[8, 16, 24], prefix_stop_exclusive=256,
        parent_origins=['A', 'S8'], parent_head='frozen C_current warmstart replayed with A and C/S8 schedules',
        warmstart='reviewed Y05 C_current checkpoint; same exact state for both arms',
        fixed_updates=2 if smoke else 1000, seed=20261028,
        learning_rate=1e-4, weight_decay=1e-4, clip_grad_norm=1.0,
        amp_initial_scale=1024.0, amp_growth_interval=2000,
        mu_anchor_delta=1e-4, checkpoint_interval=1 if smoke else 25,
        activation_checkpoint=True,
        objective_C_star='current_AbsRel + mu * normalized_anchor_delta',
        objective_CF='current_AbsRel + mean(available H1..H4 AbsRel) + mu * normalized_anchor_delta',
        future_weight=1.0, harm_penalty=False, direction_gate=False,
        sample_order='same deterministic six-slot sequence/origin round-robin order for C_star and CF',
        train_arms=list(ARMS), no_hyperparameter_grid=True,
        selection='fixed final update only; no validation-based checkpoint/gain/scene selection',
        deployment='K32/S8, gain=1; single-pulse H4 probe disables the head after the anchor',
        timing_scope='CUDA event around full train step including preparation/DIS and optimization; physical retries logged separately',
        compute_claim='equal updates only; not equal FLOPs, GT exposures, CUDA time, or total wall time',
        development_only=True, Y11_pass=False)


def gate_tolerance(repeat_macro_deltas):
    """Frozen practical margin: 1e-4 AbsRel or 3x observed repeated-inference macro spread."""
    finite = [abs(float(x)) for x in repeat_macro_deltas if x is not None and math.isfinite(float(x))]
    return max([1e-4] + [3.0 * x for x in finite])
