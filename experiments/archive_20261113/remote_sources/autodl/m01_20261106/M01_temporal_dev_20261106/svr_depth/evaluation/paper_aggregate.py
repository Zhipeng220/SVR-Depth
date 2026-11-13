"""Coverage and explicitly provisional summaries, not a paper Aggregate."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import statistics
import numpy as np

DATASETS = frozenset(('Bonn', 'TUM', 'ScanNet', 'Sintel'))


def _num(value):
    if value in ('', None):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _sequence_mean(rows, field):
    sequences = defaultdict(list)
    for row in rows:
        value = _num(row.get(field))
        if value is not None:
            sequences[(row.get('scene', ''), row['sequence'])].append(value)
    return _mean([_mean(values) for values in sequences.values()])


def summarize_coverage(frame_rows, pair_rows, *, development=False):
    """Report denominators and both common weightings without claiming authorship."""
    frames, pairs = defaultdict(list), defaultdict(list)
    for row in frame_rows:
        frames[(row['dataset'], row['method'], str(row['seed']))].append(row)
    for row in pair_rows:
        pairs[(row['dataset'], row['method'], str(row['seed']))].append(row)
    summaries = []
    for key in sorted(set(frames) | set(pairs)):
        fr, pr = frames[key], pairs[key]
        good = [r for r in fr if r.get('status') == 'ok']
        upstream = [r for r in good if r.get('upstream_depth_status') == 'ok']
        guarded_ssi = [r for r in upstream if r.get('upstream_ssi_fit_status') == 'ok']
        raw = [r for r in pr if r.get('status') == 'ok' and _num(r.get('tau5_provisional')) is not None]
        ssi = [r for r in pr if r.get('ssi_status') == 'ok' and _num(r.get('tau5_ssi_provisional')) is not None]
        candidate = [r for r in pr if r.get('status') == 'ok' and _num(r.get('tau5_candidate')) is not None]
        candidate_ssi = [r for r in pr if r.get('ssi_status') == 'ok' and _num(r.get('tau5_ssi_candidate')) is not None]
        pixels = sum(int(r.get('valid_pixels') or 0) for r in raw)
        inliers = sum(int(r.get('inliers') or 0) for r in raw)
        ssi_pixels = sum(int(r.get('ssi_valid_pixels') or 0) for r in ssi)
        ssi_inliers = sum(int(r.get('ssi_inliers') or 0) for r in ssi)
        summaries.append({
            'dataset': key[0], 'method': key[1], 'seed': key[2],
            'sequences': len({r['sequence'] for r in fr}),
            'total_frames': len(fr), 'valid_depth_frames': len(good),
            'valid_depth_pixels': sum(int(r.get('valid_pixels') or 0) for r in good),
            'upstream_metric_frames':len(upstream), 'nondegenerate_ssi_frames':len(guarded_ssi),
            'ssi_fit_status_counts':dict(Counter(str(r.get('upstream_ssi_fit_status', 'not_calculated')) for r in good)),
            'frame_status_counts': dict(Counter(str(r.get('status', '')) for r in fr)),
            'total_pairs': len(pr), 'valid_raw_pairs': len(raw), 'valid_ssi_pairs': len(ssi),
            'valid_candidate_pairs': len(candidate), 'valid_candidate_ssi_pairs': len(candidate_ssi),
            'pair_status_counts': dict(Counter(str(r.get('status', '')) for r in pr)),
            'raw_d1_frame_mean': _mean([_num(r.get('upstream_depth_d1')) for r in upstream]),
            'upstream_d1_ssi_frame_mean': _mean([_num(r.get('upstream_depth_d1_ssi')) for r in upstream]),
            'nondegenerate_d1_ssi_frame_mean': _mean([_num(r.get('upstream_depth_d1_ssi')) for r in guarded_ssi]),
            'tau5_pair_mean_provisional': _mean([_num(r.get('tau5_provisional')) for r in raw]),
            'tau5_sequence_mean_provisional': _sequence_mean(raw, 'tau5_provisional'),
            'tau5_pixel_fraction_provisional': inliers / pixels if pixels else None,
            'tau5_valid_pixels': pixels, 'tau5_inliers': inliers,
            'tau5_ssi_pair_mean_provisional': _mean([_num(r.get('tau5_ssi_provisional')) for r in ssi]),
            'tau5_ssi_sequence_mean_provisional': _sequence_mean(ssi, 'tau5_ssi_provisional'),
            'tau5_ssi_pixel_fraction_provisional': ssi_inliers / ssi_pixels if ssi_pixels else None,
            'tau5_ssi_valid_pixels': ssi_pixels, 'tau5_ssi_inliers': ssi_inliers,
            'candidate_pair_mean': _mean([_num(r.get('tau5_candidate')) for r in candidate]),
            'candidate_sequence_mean': _sequence_mean(candidate, 'tau5_candidate'),
            'candidate_pixel_fraction': (sum(int(r.get('inliers') or 0) for r in candidate) /
                                         sum(int(r.get('valid_pixels') or 0) for r in candidate)
                                         if sum(int(r.get('valid_pixels') or 0) for r in candidate) else None),
            'candidate_ssi_pair_mean': _mean([_num(r.get('tau5_ssi_candidate')) for r in candidate_ssi]),
            'candidate_ssi_sequence_mean': _sequence_mean(candidate_ssi, 'tau5_ssi_candidate'),
            'candidate_ssi_pixel_fraction': (sum(int(r.get('ssi_inliers') or 0) for r in candidate_ssi) /
                                             sum(int(r.get('ssi_valid_pixels') or 0) for r in candidate_ssi)
                                             if sum(int(r.get('ssi_valid_pixels') or 0) for r in candidate_ssi) else None),
        })
    # The inspected upstream code averages per-frame depth scores, rounds each
    # dataset to five decimals, then averages dataset results. These diagnostic
    # numbers use M01 frames/masks and do not assert full protocol equivalence.
    depth_diagnostics = []
    for method, seed in sorted({(r['method'], r['seed']) for r in summaries}):
        group = [r for r in summaries if (r['method'], r['seed']) == (method, seed)]
        if ({r['dataset'] for r in group} != DATASETS or len(group) != len(DATASETS)
                or any(r['total_frames'] != r['valid_depth_frames']
                       or r['upstream_metric_frames'] != r['total_frames']
                       or r['raw_d1_frame_mean'] is None or r['upstream_d1_ssi_frame_mean'] is None
                       for r in group)):
            continue
        depth_diagnostics.append({'method':method, 'seed':seed,
            'raw_d1':statistics.fmean(float(np.round(r['raw_d1_frame_mean'], 5)) for r in group),
            'd1_ssi':statistics.fmean(float(np.round(r['upstream_d1_ssi_frame_mean'], 5)) for r in group),
            'scope':'M01_frames_and_masks_only_not_paper_equivalent',
            'rule':'mean frames within dataset; numpy round each dataset to 5 decimals; mean four datasets'})
    temporal_diagnostics = []
    for method, seed in sorted({(r['method'], r['seed']) for r in summaries}):
        group = [r for r in summaries if (r['method'], r['seed']) == (method, seed)]
        if len(group) != len(DATASETS) or {r['dataset'] for r in group} != DATASETS:
            continue
        fields = ('tau5_pair_mean_provisional', 'tau5_pixel_fraction_provisional',
                  'tau5_sequence_mean_provisional', 'tau5_ssi_pair_mean_provisional',
                  'tau5_ssi_pixel_fraction_provisional', 'tau5_ssi_sequence_mean_provisional',
                  'candidate_pair_mean', 'candidate_sequence_mean', 'candidate_pixel_fraction',
                  'candidate_ssi_pair_mean', 'candidate_ssi_sequence_mean', 'candidate_ssi_pixel_fraction')
        values = {field: (statistics.fmean(r[field] for r in group)
                           if all(r[field] is not None for r in group) else None) for field in fields}
        temporal_diagnostics.append({'method': method, 'seed': seed, **values,
            'scope': 'equal_dataset_candidate_only_not_author_verified'})
    reasons = ['author_temporal_equivalence_unverified', 'author_temporal_aggregation_rule_unverified',
               'paper_preprocessing_frame_selection_and_split_unverified']
    if development:
        reasons.append('development_frame_truncation')
    for method in ('A', 'P_multi'):
        if {r['dataset'] for r in summaries if r['method'] == method} != DATASETS:
            reasons.append(f'incomplete_dataset_coverage_{method}')
    if not pair_rows or any(r.get('status') != 'ok' or r.get('ssi_status') != 'ok' for r in pair_rows):
        reasons.append('temporal_pair_coverage_incomplete')
    return {'scope': 'coverage_and_provisional_summaries_only', 'development_only': bool(development),
            'formal_aggregate_ready': False, 'formal_aggregate': None, 'blocking_reasons': reasons,
            'aggregation_note': 'Both pair-weighted and valid-pixel-weighted values are shown; neither is asserted to be the authors rule.',
            'upstream_depth_equal_dataset_diagnostic':depth_diagnostics,
            'temporal_equal_dataset_diagnostics': temporal_diagnostics,
            'datasets': summaries}


def require_formal_aggregate(report=None, *, evidence=None, evidence_base=None):
    """Return only tables produced by the evidence-gated final report builder."""
    if (not isinstance(report, dict) or report.get('formal_aggregate_ready') is not True
            or report.get('scope') != 'reviewer_attested_formal' or report.get('blocking_reasons')
            or not isinstance(report.get('tables'), list) or not evidence or evidence_base is None):
        raise RuntimeError('Formal paper Aggregate is unavailable until coverage and reviewed evidence are verified')
    from svr_depth.evaluation.m01_paper_report import verify_evidence
    if verify_evidence(evidence, evidence_base, report.get('input_sha256')):
        raise RuntimeError('Formal paper Aggregate evidence is missing, changed or bound to different inputs')
    return report['tables']
