"""Traceable four-dataset result tables; scientific equivalence is separately audited."""
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import statistics

DATASETS = frozenset(('Bonn', 'TUM', 'ScanNet', 'Sintel'))
REQUIRED_METHODS = frozenset(('A_native', 'A', 'P_multi'))
REQUIRED_CHECKS = ('paper_protocol_verified', 'flow_contract_verified', 'camera_contract_verified',
                   'pose_contract_verified', 'tau5_oracle_passed', 'ssi_equivalence_passed',
                   'aggregate_verified', 'all_dataset_coverage_passed', 'native_baseline_verified',
                   'tau5_candidate_validated', 'tau5_author_equivalence_verified')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_evidence(manifest, base, input_hashes):
    """Check signed-off artifacts exist, match hashes and bind the exact inputs.

    This validates artifact integrity and scope, not the scientific truth of an
    assertion. A reviewer remains responsible for inspecting the artifacts.
    """
    if not isinstance(input_hashes, dict) or not input_hashes or not isinstance(manifest, dict) or manifest.get('input_sha256') != input_hashes:
        return ['evidence_input_hash_mismatch']
    checks = manifest.get('checks', {})
    problems = []
    for name in REQUIRED_CHECKS:
        item = checks.get(name, {}) if isinstance(checks, dict) else {}
        if not isinstance(item, dict) or item.get('passed') is not True or not item.get('reviewer'):
            problems.append('unverified_' + name); continue
        artifact, expected = item.get('artifact'), item.get('sha256')
        if not isinstance(artifact, str) or not isinstance(expected, str) or len(expected) != 64:
            problems.append('invalid_evidence_' + name); continue
        path = Path(artifact).expanduser()
        path = path if path.is_absolute() else Path(base) / path
        try:
            actual = sha256_file(path)
        except OSError:
            problems.append('missing_evidence_' + name); continue
        if actual.lower() != expected.lower():
            problems.append('evidence_hash_mismatch_' + name)
    return problems


def _number(value):
    try: value = float(value)
    except (TypeError, ValueError): return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def _mean(rows, field):
    vals = [_number(r.get(field)) for r in rows]
    return statistics.fmean(vals) if vals and all(x is not None for x in vals) else None


def _temporal(rows, field, rule, prefix):
    if not rows or any(_number(r.get(field)) is None for r in rows): return None
    if rule == 'equal_pair': return _mean(rows, field)
    if rule == 'equal_sequence':
        groups = defaultdict(list)
        for row in rows: groups[(row.get('scene'), row['sequence'])].append(row)
        return statistics.fmean(_mean(group, field) for group in groups.values())
    if rule == 'pixel_weighted':
        try:
            nums = [(int(r[prefix + 'inliers']), int(r[prefix + 'valid_pixels'])) for r in rows]
        except (ValueError, KeyError, TypeError): return None
        if any(n <= 0 or i < 0 or i > n or abs(i/n - float(r[field])) > 1e-6
               for (i, n), r in zip(nums, rows)): return None
        return sum(i for i, _ in nums) / sum(n for _, n in nums)
    raise ValueError('Choose equal_pair, equal_sequence or pixel_weighted explicitly')


def build_report(frame_rows, pair_rows, *, temporal_rule, declared_methods=REQUIRED_METHODS):
    if temporal_rule not in ('equal_pair', 'equal_sequence', 'pixel_weighted'):
        raise ValueError('Temporal aggregation rule must be explicitly selected')
    frames, pairs = defaultdict(list), defaultdict(list)
    problems = []
    for row in frame_rows: frames[(row['dataset'], row['method'], str(row['seed']))].append(row)
    for row in pair_rows: pairs[(row['dataset'], row['method'], str(row['seed']))].append(row)
    present = {k[1] for k in frames}
    for method in sorted(set(declared_methods) - present): problems.append('missing_method:' + method)
    identities = {(method, seed) for _, method, seed in frames}
    tables = []
    for method, seed in sorted(identities):
        datasets = []
        for dataset in sorted(DATASETS):
            fr, pr = frames[(dataset, method, seed)], pairs[(dataset, method, seed)]
            if not fr:
                problems.append(f'missing_frames:{dataset}:{method}:{seed}'); continue
            frame_ids = [(r['scene'], r['sequence'], r['index']) for r in fr]
            if len(set(frame_ids)) != len(frame_ids): problems.append(f'duplicate_frames:{dataset}:{method}:{seed}')
            if any(r.get('status') != 'ok' or r.get('upstream_depth_status') != 'ok' for r in fr):
                problems.append(f'invalid_depth_coverage:{dataset}:{method}:{seed}')
            sequences = defaultdict(list)
            for r in fr: sequences[(r['scene'], r['sequence'])].append(int(r['index']))
            expected_pairs = {(scene, seq, str(a), str(b)) for (scene, seq), indices in sequences.items()
                              for a, b in zip(sorted(indices), sorted(indices)[1:]) if b == a + 1}
            actual_pairs = [(r['scene'], r['sequence'], str(r['prev_index']), str(r['curr_index'])) for r in pr]
            if len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != expected_pairs:
                problems.append(f'incomplete_pair_coverage:{dataset}:{method}:{seed}')
            if any(r.get('status') != 'ok' or r.get('ssi_status') != 'ok'
                   or r.get('temporal_protocol') != 'paper_tau5_candidate_v1' for r in pr):
                problems.append(f'invalid_temporal_coverage:{dataset}:{method}:{seed}')
            metrics = {'delta1': _mean(fr, 'upstream_depth_d1'),
                       'delta1_ssi': _mean(fr, 'upstream_depth_d1_ssi'),
                       'tau5': _temporal(pr, 'tau5_candidate', temporal_rule, ''),
                       'tau5_ssi': _temporal(pr, 'tau5_ssi_candidate', temporal_rule, 'ssi_')}
            if any(x is None for x in metrics.values()): problems.append(f'missing_metric:{dataset}:{method}:{seed}')
            datasets.append({'dataset': dataset, **metrics, 'frames': len(fr), 'pairs': len(pr),
                             'valid_pixels': sum(int(r.get('valid_pixels') or 0) for r in pr)})
        aggregate = None
        if len(datasets) == 4 and all(all(d[m] is not None for m in ('delta1','delta1_ssi','tau5','tau5_ssi')) for d in datasets):
            aggregate = {m: statistics.fmean(round(d[m], 5) if m in ('delta1','delta1_ssi') else d[m] for d in datasets)
                         for m in ('delta1','delta1_ssi','tau5','tau5_ssi')}
        tables.append({'method':method, 'seed':seed, 'datasets':datasets, 'aggregate':aggregate})
    # Fairness: same source identities and scoring masks for all methods/seeds.
    for dataset in DATASETS:
        groups = [(key, rows) for key, rows in frames.items() if key[0] == dataset and rows]
        if not groups: continue
        def signatures(rows):
            return {(r.get('sequence'), r.get('source_frame_id'), r.get('source_frame_index'),
                     r.get('rgb_sha256'), r.get('gt_sha256'), r.get('mask_sha256')) for r in rows}
        reference = signatures(groups[0][1])
        if any(signatures(rows) != reference for _, rows in groups[1:]):
            problems.append('frame_or_mask_mismatch:' + dataset)
        if any(not r.get('rgb_sha256') or not r.get('gt_sha256') or not r.get('mask_sha256') or not r.get('source_frame_id')
               for _, rows in groups for r in rows):
            problems.append('missing_frame_provenance:' + dataset)
        pair_groups = [(key, rows) for key, rows in pairs.items() if key[0] == dataset and rows]
        fields = ('flow_bwd_sha256', 'flow_bwd_mask_sha256', 'prev_gt_mask_sha256', 'curr_gt_mask_sha256',
                  'prev_pose_sha256', 'curr_pose_sha256', 'prev_intrinsics_sha256', 'curr_intrinsics_sha256')
        if pair_groups:
            def pair_signatures(rows):
                return {(r.get('sequence'), str(r.get('prev_index')), str(r.get('curr_index')),
                         *(r.get(field) for field in fields)) for r in rows}
            reference = pair_signatures(pair_groups[0][1])
            if any(pair_signatures(rows) != reference for _, rows in pair_groups[1:]):
                problems.append('temporal_input_mismatch:' + dataset)
            if any(not r.get(field) for _, rows in pair_groups for r in rows for field in fields):
                problems.append('missing_temporal_provenance:' + dataset)
    return {'scope':'candidate_requires_independent_protocol_review', 'temporal_rule': temporal_rule,
            'data_blocking_reasons': sorted(set(problems)), 'tables': tables}
