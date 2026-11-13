#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select top-K candidate anchors by predictor score, without reading target labels for ranking.

The ranking pass is a retrospective diagnostic, not a deployable online policy:
it uses the full score list from a no-writeback replay to choose K action frames.
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def truthy(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--score-csv", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--model", choices=["learned_state_only", "motion_plus_learned"], required=True)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--candidate-stride", type=int, default=4)
    ap.add_argument("--candidate-offset", type=int, default=1)
    ap.add_argument("--fixed-interval", type=int, default=8)
    ap.add_argument("--min-first-frame", type=int, default=5)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--print-frames", action="store_true")
    args = ap.parse_args()

    ds = pd.read_csv(args.dataset)
    ds["sequence"] = ds["sequence"].astype(str).str.lower()
    ds["anchor_frame"] = pd.to_numeric(ds["anchor_frame"], errors="coerce")
    ds["horizon_H"] = pd.to_numeric(ds["horizon_H"], errors="coerce")
    ds = ds[(ds.sequence == args.sequence.lower()) &
            (ds.horizon_H == args.horizon) & truthy(ds["eligible"])].copy()
    labeled = sorted({
        int(a) for a in ds["anchor_frame"].dropna().tolist()
        if int(a) >= args.min_first_frame and
        (int(a) - args.candidate_offset) % args.candidate_stride == 0
    })
    fixed = [
        a for a in labeled
        if (a - args.candidate_offset) % args.fixed_interval == 0
    ]
    k = len(fixed)
    if k == 0:
        raise SystemExit(f"No eligible fixed-budget anchors for {args.sequence}; inspect the dataset.")
    scores = pd.read_csv(args.score_csv)
    scores["sequence"] = scores["sequence"].astype(str).str.lower()
    scores["frame_index"] = pd.to_numeric(scores["frame_index"], errors="coerce")
    scores["predictor_score"] = pd.to_numeric(scores["predictor_score"], errors="coerce")
    scores = scores[(scores.sequence == args.sequence.lower()) &
                    (scores.policy == args.model) &
                    (scores.frame_index.isin(labeled)) &
                    np.isfinite(scores["predictor_score"])].copy()
    scores["frame_index"] = scores["frame_index"].astype(int)
    if scores.frame_index.nunique() < k:
        raise SystemExit(f"Only {scores.frame_index.nunique()} finite score rows for budget K={k}.")
    ranked = scores.sort_values(["predictor_score", "frame_index"], ascending=[False, True])
    selected = ranked.head(k)
    selected_frames = selected.frame_index.astype(int).tolist()

    payload = {
        "sequence": args.sequence.lower(),
        "model": args.model,
        "horizon_H": args.horizon,
        "eligible_labeled_candidate_frames": labeled,
        "fixed_baseline_frames_with_complete_label": fixed,
        "action_budget_K": k,
        "selected_topk_frames_by_score_only": selected_frames,
        "selected_scores": [
            {"frame_index": int(r.frame_index), "predictor_score": float(r.predictor_score)}
            for r in selected.itertuples()
        ],
        "score_rows_used": int(len(scores)),
        "selection_uses_target_labels": False,
        "interpretation": "Retrospective score-ranked fixed-budget diagnostic, not causal online scheduling. Targets are not read by the selection step.",
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.print_frames:
        print(",".join(map(str, selected_frames)))
    else:
        print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
