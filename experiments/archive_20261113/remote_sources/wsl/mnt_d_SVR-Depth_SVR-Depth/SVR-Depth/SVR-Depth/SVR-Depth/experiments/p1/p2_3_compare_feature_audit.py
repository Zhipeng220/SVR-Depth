#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare online pre-action features against P2.2.2 offline feature rows."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

META_COLUMNS = {
    "sequence", "anchor_frame", "horizon_H",
    "G_state_absrel_discounted_per_valid_frame", "G_state_absrel_discounted",
    "future_frames_requested", "future_frames_with_gt", "future_gt_coverage_ratio",
    "future_frames_evaluated", "future_metric_pair_coverage_ratio", "anchor_gt_available",
    "gamma", "future_policy", "discount_policy", "anchor_GT_available_audit_only",
    "valid_discount_weight_sum", "finite_future_metric_pairs_recounted",
    "target_discount_weight_normalized", "restart_steps_ignored_within_horizon",
    "eligible", "exclusion_reason",
    "policy", "feature_timing", "last_keyframe_before_action",
    "distance_kf_before_action", "anchor_rgb_timestamp", "frame_index",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", required=True)
    ap.add_argument("--dataset", default="outputs/p2_2_2/p2_2_2_valid_dataset.csv")
    ap.add_argument("--sequence", default="crowd2")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    online = pd.read_csv(args.online)
    dataset = pd.read_csv(args.dataset)
    online["sequence"] = online["sequence"].astype(str).str.lower()
    dataset["sequence"] = dataset["sequence"].astype(str).str.lower()
    online["anchor_frame"] = pd.to_numeric(online["anchor_frame"], errors="coerce")
    dataset["anchor_frame"] = pd.to_numeric(dataset["anchor_frame"], errors="coerce")
    dataset["horizon_H"] = pd.to_numeric(dataset["horizon_H"], errors="coerce")

    online = online[online["sequence"] == args.sequence.lower()].copy()
    train = dataset[
        (dataset["sequence"] == args.sequence.lower())
        & (dataset["horizon_H"] == args.horizon)
    ].copy()

    keys = ["sequence", "anchor_frame"]
    merged = online.merge(train, on=keys, how="outer", suffixes=("_online", "_offline"), indicator=True)
    matched = merged[merged["_merge"] == "both"].copy()
    online_only = merged[merged["_merge"] == "left_only"][keys].copy()
    offline_only = merged[merged["_merge"] == "right_only"][keys].copy()

    shared = []
    for col in online.columns:
        if col in META_COLUMNS or col not in train.columns:
            continue
        if pd.api.types.is_numeric_dtype(online[col]) and pd.api.types.is_numeric_dtype(train[col]):
            shared.append(col)

    summaries = []
    details = []
    for col in shared:
        a = pd.to_numeric(matched[f"{col}_online"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(matched[f"{col}_offline"], errors="coerce").to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        aa, bb = a[finite], b[finite]
        if len(aa):
            diff = aa - bb
            close = np.isclose(aa, bb, rtol=args.rtol, atol=args.atol)
            corr = float(np.corrcoef(aa, bb)[0, 1]) if len(aa) > 1 and np.std(aa) > 0 and np.std(bb) > 0 else float("nan")
            summaries.append({
                "feature": col, "n_matched_finite": int(len(aa)),
                "online_mean": float(np.mean(aa)), "offline_mean": float(np.mean(bb)),
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "max_abs_diff": float(np.max(np.abs(diff))),
                "rmse_diff": float(np.sqrt(np.mean(diff**2))),
                "pearson_r": corr, "n_allclose": int(np.sum(close)),
                "allclose_at_tolerance": bool(np.all(close)),
            })
            for (_, row), av, bv in zip(matched.iterrows(), a, b):
                if np.isfinite(av) and np.isfinite(bv):
                    details.append({
                        "sequence": row["sequence"], "anchor_frame": row["anchor_frame"],
                        "feature": col, "online_value": av, "offline_value": bv,
                        "abs_diff": abs(av-bv), "allclose": bool(np.isclose(av,bv,rtol=args.rtol,atol=args.atol))
                    })
        else:
            summaries.append({"feature": col, "n_matched_finite": 0, "allclose_at_tolerance": False})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    detail_df = pd.DataFrame(details)
    summary_df.to_csv(out / "p2_3_feature_consistency_summary.csv", index=False)
    detail_df.to_csv(out / "p2_3_feature_consistency_detail.csv", index=False)
    online_only.to_csv(out / "p2_3_online_only_anchors.csv", index=False)
    offline_only.to_csv(out / "p2_3_offline_only_anchors.csv", index=False)

    overall = {
        "sequence": args.sequence,
        "horizon": args.horizon,
        "online_rows": int(len(online)),
        "offline_rows": int(len(train)),
        "matched_anchor_rows": int(len(matched)),
        "online_only_anchors": online_only["anchor_frame"].dropna().astype(int).tolist(),
        "offline_only_anchors": offline_only["anchor_frame"].dropna().astype(int).tolist(),
        "shared_numeric_feature_count": int(len(shared)),
        "all_features_allclose": bool(len(summary_df) > 0 and summary_df["allclose_at_tolerance"].fillna(False).all()),
        "rtol": args.rtol,
        "atol": args.atol,
        "interpretation": "Compare pre-action online features on auto replay with P2.2.2 rows at same sequence/anchor and H. Features are inputs only; target/GT-derived columns are excluded."
    }
    (out / "p2_3_feature_consistency_metadata.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2))
    if len(summary_df):
        print("\nLargest feature discrepancies:")
        print(summary_df.sort_values("max_abs_diff", ascending=False).head(20).to_string(index=False))
    if len(matched) == 0:
        raise SystemExit("FAIL: no matching anchors; check sequence/anchor indexing.")
    if not overall["all_features_allclose"]:
        print("\nWARNING: at least one numeric feature differs beyond tolerance. Inspect summary/detail CSV.")
    else:
        print("\nPASS: all shared numeric input features match within tolerance.")

if __name__ == "__main__":
    main()
