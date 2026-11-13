#!/usr/bin/env python3
"""Produce auditable M01 tables from a completed quality run and evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svr_depth.evaluation.m01_paper_report import build_report, sha256_file, verify_evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--temporal-rule', required=True, choices=('equal_pair', 'equal_sequence', 'pixel_weighted'))
    p.add_argument('--evidence', type=Path, help='Reviewer-attested evidence manifest bound to these exact result files')
    p.add_argument('--require-formal', action='store_true', help='Fail unless coverage and evidence checks pass')
    a = p.parse_args()
    run = a.run_dir.expanduser().resolve(strict=True)
    paths = {key: run / name for key, name in {'frames':'frame_metrics.csv', 'pairs':'temporal_metrics.csv',
                                                'identity':'run_identity.json'}.items()}
    identity = json.loads(paths['identity'].read_text(encoding='utf-8-sig'))
    with paths['frames'].open(encoding='utf-8-sig', newline='') as f: frames = list(csv.DictReader(f))
    with paths['pairs'].open(encoding='utf-8-sig', newline='') as f: pairs = list(csv.DictReader(f))
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    report = build_report(frames, pairs, temporal_rule=a.temporal_rule)
    problems = report['data_blocking_reasons'][:]
    if identity.get('status') != 'complete' or identity.get('mode') != 'quality': problems.append('run_not_complete_quality')
    if identity.get('run_scope') == 'development_only_truncated': problems.append('development_run')
    cfg = identity.get('resolved_config', {})
    if cfg.get('formal_paper_result') is not True or identity.get('run_scope') != 'formal_requested_awaiting_result_review':
        problems.append('run_not_admitted_as_formal_candidate')
    if cfg.get('paper_protocol_status') != 'reviewed': problems.append('paper_protocol_not_reviewed')
    if cfg.get('preprocessing_protocol') != 'upstream_verified': problems.append('preprocessing_not_verified')
    if cfg.get('camera_gt') is not False: problems.append('camera_gt_not_fixed_false')
    if not cfg.get('hash_hdf5_content'): problems.append('hdf5_content_not_hashed')
    configured = {(str(m.get('method')), str(m.get('seed'))) for m in cfg.get('methods', [])}
    observed = {(r['method'], str(r['seed'])) for r in frames}
    if configured != observed: problems.append('configured_and_observed_methods_differ')
    seqs = cfg.get('sequences', [])
    declared = {(s.get('dataset'), s.get('sequence')) for s in seqs}
    if not seqs or any(s.get('data_role') != 'test' for s in seqs): problems.append('test_roles_not_frozen')
    for method, seed in observed:
        rows = [r for r in frames if (r['method'], str(r['seed'])) == (method, seed)]
        if {(r['dataset'], r['sequence']) for r in rows} != declared:
            problems.append('configured_and_observed_sequences_differ')
        for spec in seqs:
            group = sorted((r for r in rows if r['dataset'] == spec.get('dataset') and r['sequence'] == spec.get('sequence')),
                           key=lambda r:int(r['index']))
            ids = [r['source_frame_id'] for r in group]
            actual = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
            if not spec.get('source_frame_ids_sha256') or spec['source_frame_ids_sha256'] != actual:
                problems.append('frozen_frame_manifest_mismatch')
    if a.evidence:
        path = a.evidence.expanduser().resolve(strict=True)
        evidence = json.loads(path.read_text(encoding='utf-8-sig'))
        problems.extend(verify_evidence(evidence, path.parent, hashes))
        if evidence.get('temporal_rule') != a.temporal_rule: problems.append('evidence_temporal_rule_mismatch')
    else:
        problems.append('missing_reviewed_evidence')
    report.update(input_sha256=hashes, blocking_reasons=sorted(set(problems)),
                  formal_aggregate_ready=not problems,
                  evidence_scope='Artifact hash and scope checks do not independently establish scientific validity.')
    report['scope'] = 'reviewer_attested_formal' if not problems else 'candidate_not_formal'
    report['temporal_metric_identity'] = ('author_equivalent' if not problems else
        'project_defined_candidate' if any(r.get('tau5_candidate') not in (None, '') for r in pairs) else 'unavailable')
    if a.require_formal and problems:
        raise RuntimeError('Formal reporting blocked: ' + ', '.join(sorted(set(problems))))
    out = a.out_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=False)
    (out / 'm01_paper_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    rows = []
    for table in report['tables']:
        for d in table['datasets']:
            rows.append({'method':table['method'], 'seed':table['seed'], **d})
        if table['aggregate'] is not None:
            rows.append({'method':table['method'], 'seed':table['seed'], 'dataset':'Aggregate', **table['aggregate']})
    columns = ['method','seed','dataset','delta1','delta1_ssi','tau5','tau5_ssi','frames','pairs','valid_pixels']
    with (out/'m01_paper_table.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    lines = ['# M01 results', '', 'Status: ' + report['scope'], '',
             'Temporal rule: ' + a.temporal_rule, '',
             '| Method | Seed | Dataset | δ1 | δ1 SSI | τ5 | τ5 SSI | Frames | Pairs |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        def fmt(name): return '—' if row.get(name) is None else f"{100 * row[name]:.3f}%"
        lines.append(f"| {row['method']} | {row['seed']} | {row['dataset']} | {fmt('delta1')} | {fmt('delta1_ssi')} | {fmt('tau5')} | {fmt('tau5_ssi')} | {row.get('frames', '—')} | {row.get('pairs', '—')} |")
    lines += ['', 'Blocking reasons: ' + (', '.join(report['blocking_reasons']) or 'none'),
              '', report['evidence_scope'], '']
    (out/'m01_paper_table.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Report:', out, 'scope:', report['scope'])


if __name__ == '__main__': main()
