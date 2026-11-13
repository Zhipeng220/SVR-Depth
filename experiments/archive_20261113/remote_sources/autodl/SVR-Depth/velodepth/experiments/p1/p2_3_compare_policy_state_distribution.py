#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare feature distributions across online policy trajectories and P2.2.2 rows."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

META = {
    "sequence", "anchor_frame", "horizon_H", "policy", "feature_timing",
    "last_keyframe_before_action", "distance_kf_before_action",
    "scheduler_current_t", "scheduler_first_keyframe",
    "G_state_absrel_discounted_per_valid_frame", "G_state_absrel_discounted",
    "future_frames_requested", "future_frames_with_gt", "future_gt_coverage_ratio",
    "future_frames_evaluated", "future_metric_pair_coverage_ratio",
    "anchor_gt_available", "gamma", "future_policy", "discount_policy",
    "anchor_GT_available_audit_only", "valid_discount_weight_sum",
    "finite_future_metric_pairs_recounted", "target_discount_weight_normalized",
    "restart_steps_ignored_within_horizon", "eligible", "exclusion_reason",
}

def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])

def smd(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled <= 1e-12:
        return 0.0 if abs(np.mean(a)-np.mean(b)) <= 1e-12 else float("inf")
    return float((np.mean(a)-np.mean(b))/pooled)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", required=True)
    ap.add_argument("--frame-metrics", required=True)
    ap.add_argument("--dataset", default="outputs/p2_2_2/p2_2_2_valid_dataset.csv")
    ap.add_argument("--sequence", default="crowd2")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    online = pd.read_csv(args.online)
    metrics = pd.read_csv(args.frame_metrics)
    dataset = pd.read_csv(args.dataset)
    for df in (online, metrics, dataset):
        if "sequence" in df:
            df["sequence"] = df["sequence"].astype(str).str.lower()
    online["anchor_frame"] = pd.to_numeric(online["anchor_frame"], errors="coerce")
    dataset["anchor_frame"] = pd.to_numeric(dataset["anchor_frame"], errors="coerce")
    dataset["horizon_H"] = pd.to_numeric(dataset["horizon_H"], errors="coerce")
    seq = args.sequence.lower()
    online = online[online.sequence == seq].copy()
    ds = dataset[(dataset.sequence == seq) & (dataset.horizon_H == args.horizon)].copy()
    if online.empty:
        raise SystemExit("No online audit rows for requested sequence.")
    if ds.empty:
        raise SystemExit("No offline rows for requested sequence/horizon.")

    feature_cols = [
        c for c in online.columns
        if c not in META and c in ds.columns
        and pd.api.types.is_numeric_dtype(online[c])
        and pd.api.types.is_numeric_dtype(ds[c])
    ]
    policies = sorted(online.policy.dropna().astype(str).unique().tolist())
    if "auto" not in policies:
        raise SystemExit("The online audit must include policy=auto as the no-extra-refresh reference.")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    per_feature = []
    anchor_detail = []
    ds_idx = ds.set_index(["sequence", "anchor_frame"])
    auto = online[online.policy == "auto"].set_index(["sequence", "anchor_frame"])
    for policy in policies:
        pol = online[online.policy == policy].set_index(["sequence", "anchor_frame"])
        common = pol.index.intersection(auto.index).intersection(ds_idx.index)
        if len(common) == 0:
            continue
        for feature in feature_cols:
            p = pd.to_numeric(pol.loc[common, feature], errors="coerce").to_numpy(float)
            a = pd.to_numeric(auto.loc[common, feature], errors="coerce").to_numpy(float)
            d = pd.to_numeric(ds_idx.loc[common, feature], errors="coerce").to_numpy(float)
            valid = np.isfinite(p) & np.isfinite(a) & np.isfinite(d)
            if valid.sum() == 0:
                continue
            p, a, d = p[valid], a[valid], d[valid]
            delta_auto = p-a
            delta_off = p-d
            per_feature.append({
                "policy": policy, "feature": feature, "n_matched_anchors": int(len(p)),
                "policy_mean": float(np.mean(p)), "policy_std": float(np.std(p, ddof=1)) if len(p)>1 else float("nan"),
                "auto_mean_same_anchors": float(np.mean(a)), "auto_std_same_anchors": float(np.std(a, ddof=1)) if len(a)>1 else float("nan"),
                "offline_h16_mean_same_anchors": float(np.mean(d)), "offline_h16_std_same_anchors": float(np.std(d, ddof=1)) if len(d)>1 else float("nan"),
                "mean_delta_vs_auto": float(np.mean(delta_auto)), "max_abs_delta_vs_auto": float(np.max(np.abs(delta_auto))),
                "smd_vs_auto": smd(p,a), "corr_vs_auto": corr(p,a),
                "mean_abs_delta_vs_offline": float(np.mean(np.abs(delta_off))),
                "max_abs_delta_vs_offline": float(np.max(np.abs(delta_off))),
                "smd_vs_offline_h16": smd(p,d),
                "fraction_changed_vs_auto": float(np.mean(np.abs(delta_auto)>1e-8)),
            })
            for (s, anchor), pv, av, dv in zip(common, p, a, d):
                anchor_detail.append({
                    "policy": policy, "sequence": s, "anchor_frame": anchor,
                    "feature": feature, "policy_value": pv, "auto_value": av,
                    "offline_h16_value": dv, "delta_vs_auto": pv-av,
                    "delta_vs_offline": pv-dv,
                })
    summary = pd.DataFrame(per_feature)
    detail = pd.DataFrame(anchor_detail)
    summary.to_csv(out/"p2_3_policy_feature_distribution_summary.csv", index=False)
    detail.to_csv(out/"p2_3_policy_feature_distribution_detail.csv", index=False)

    m = metrics[metrics.sequence == seq].copy()
    action_summary = m.groupby(["policy","action"], dropna=False).size().rename("count").reset_index()
    action_summary.to_csv(out/"p2_3_policy_action_counts.csv", index=False)
    state_rows = []
    for policy, g in online.groupby("policy"):
        dist = pd.to_numeric(g["distance_kf_before_action"], errors="coerce")
        state_rows.append({
            "policy": policy, "candidate_rows": int(len(g)),
            "distance_kf_mean": float(dist.mean()), "distance_kf_median": float(dist.median()),
            "distance_kf_min": float(dist.min()), "distance_kf_max": float(dist.max()),
            "distance_kf_unique": ";".join(map(str, sorted(dist.dropna().unique().tolist()))),
            "last_keyframe_unique": ";".join(map(str, sorted(pd.to_numeric(g["last_keyframe_before_action"], errors="coerce").dropna().unique().tolist()))),
        })
    pd.DataFrame(state_rows).to_csv(out/"p2_3_policy_state_age_summary.csv", index=False)

    meta = {
        "sequence": seq, "horizon": args.horizon,
        "online_feature_rows": int(len(online)), "policies": policies,
        "candidate_anchors_per_policy": {
            str(k): sorted(pd.to_numeric(v["anchor_frame"], errors="coerce").dropna().astype(int).unique().tolist())
            for k,v in online.groupby("policy")
        },
        "offline_h16_rows": int(len(ds)),
        "shared_input_feature_count": int(len(feature_cols)),
        "interpretation": "Feature distributions are compared at the same candidate anchor before that policy's current action. For fixed8/candidate_fixed, later anchors reflect prior refreshes. These are policy-conditioned covariate-shift diagnostics, not causal proof or a quality result.",
        "caveats": [
            "P2.2.2 H=16 feature rows come from the offline reference trajectory; they are not refreshed-policy training data.",
            "Neighboring anchors are temporally dependent.",
            "Standardized mean differences are descriptive with only 18 matched H=16 anchors."
        ]
    }
    (out/"p2_3_policy_feature_distribution_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("\\nLargest candidate_fixed vs auto shifts:")
    if not summary.empty:
        s = summary[summary.policy == "candidate_fixed"].copy()
        s["abs_smd"] = s["smd_vs_auto"].abs()
        print(s.sort_values("abs_smd", ascending=False).head(20)[
            ["feature","mean_delta_vs_auto","smd_vs_auto","fraction_changed_vs_auto"]
        ].to_string(index=False))
    if set(policies) != {"auto","candidate_fixed","fixed8"}:
        print("WARNING: expected policies are auto,candidate_fixed,fixed8; check the replay command.")

if __name__ == "__main__":
    main()
