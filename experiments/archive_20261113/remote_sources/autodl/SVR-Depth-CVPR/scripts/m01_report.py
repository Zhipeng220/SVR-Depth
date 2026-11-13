#!/usr/bin/env python3
"""Rebuild M01 summaries from an existing quality or timing results directory."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svr_depth.evaluation.m01_runner import _csv, _json
from svr_depth.evaluation.aggregate import summarize_frames, paired_risk, latency_summary, risk_hierarchy, latency_hierarchy


def read(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out-dir', required=True, help='new report directory')
    p.add_argument('--bootstrap-samples', type=int, default=2000)
    args = p.parse_args()
    run = Path(args.run_dir).resolve(strict=True)
    ident = json.loads((run/'run_identity.json').read_text(encoding='utf8'))
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if ident['mode'] == 'quality':
        bootstrap = args.bootstrap_samples if ident['status'] == 'complete' and ident.get('run_scope') != 'development_only_truncated' else 0
        rows = read(run/'frame_metrics.csv')
        seq, scene, dataset = summarize_frames(rows, bootstrap)
        _csv(out/'sequence_summary.csv', seq)
        _csv(out/'scene_summary.csv', scene)
        _json(out/'dataset_summary.json', {'source_run_status':ident['status'], 'run_scope':ident.get('run_scope'), 'datasets':dataset})
        risks = paired_risk(rows)
        _csv(out/'risk_summary.csv', risks)
        scenes, datasets, comparisons = risk_hierarchy(risks, bootstrap)
        _csv(out/'risk_scene_summary.csv', scenes)
        _csv(out/'risk_dataset_summary.csv', datasets)
        _json(out/'paired_comparison.json', {'source_run_status':ident['status'], 'run_scope':ident.get('run_scope'), 'comparisons':comparisons})
    elif ident['mode'] == 'timing':
        timing = latency_summary(read(run/'timing_frames.csv'))
        _csv(out/'latency_summary.csv', timing)
        _csv(out/'latency_dataset_summary.csv', latency_hierarchy(timing))
    else:
        raise ValueError(f"Unsupported run mode: {ident['mode']}")
    print(f'Report written to {out}; source status={ident["status"]}')


if __name__ == '__main__':
    main()
