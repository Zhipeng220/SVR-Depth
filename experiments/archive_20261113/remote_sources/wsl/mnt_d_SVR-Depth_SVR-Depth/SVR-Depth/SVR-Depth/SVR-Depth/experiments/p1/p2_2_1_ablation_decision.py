#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVR-Depth P2.2.1: feature ablation + decision usefulness, using existing P1.5 CSVs.

Run from the VeloDepth repo:
python experiments/p1/p2_2_1_ablation_decision.py \
  --input-dir outputs/p2_2_review \
  --out-dir outputs/p2_2_1

No GPU/model inference is required. Preprocessing is fitted within each
leave-one-sequence-out fold. Anchor GT availability is audited but is NOT a
hard exclusion when the future target is valid. Horizons whose forced-
propagation window crosses a logged ignored restart request are excluded.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEQUENCES = ("crowd2", "balloon2", "crowd3")
ALL_HORIZONS = (1, 4, 8, 16)
TARGET = "G_state_absrel_discounted_per_valid_frame"

# Small, interpretable feature groups. IDs/timestamps and scheduler_current_t
# are deliberately excluded from predictors to avoid trivial time/index leakage.
MOTION = [
    "scheduler_distance_kf",
    "scheduler_flow_magnitude",
    "scheduler_total_luma_change",
    "image_absdiff_mean_prev",
    "image_absdiff_p90_prev",
]
GEOMETRY = [
    "state_radius_teacher_mean",
    "state_radius_teacher_std",
    "state_radius_teacher_q10",
    "state_radius_teacher_q50",
    "state_radius_teacher_q90",
    "state_rays_base_mean",
    "state_rays_base_std",
    "state_rays_base_q10",
    "state_rays_base_q50",
    "state_rays_base_q90",
]
CONFIDENCE = [
    "state_confidence_teacher_mean",
    "state_confidence_teacher_std",
    "state_confidence_teacher_q10",
    "state_confidence_teacher_q50",
    "state_confidence_teacher_q90",
]
LEARNED_STATE = [
    f"state_features_teacher_{i}_{stat}"
    for i in range(4) for stat in ("mean", "std")
]
MODEL_FEATURES = {
    "mean_baseline": [],
    "motion_only": MOTION,
    "geometry_only": GEOMETRY,
    "confidence_only": CONFIDENCE,
    "geometry_confidence": GEOMETRY + CONFIDENCE,
    "learned_state_only": LEARNED_STATE,
    "state_motion_full": GEOMETRY + CONFIDENCE + LEARNED_STATE + MOTION,
}


def as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def truthy(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def read_csv(path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_restart_steps(input_dir, sequence):
    path = input_dir / f"{sequence}_multihorizon_frame_metrics.csv"
    df = read_csv(path, required=False)
    if df.empty or not {"anchor_frame", "future_step"}.issubset(df.columns):
        return {}
    flags = [c for c in ("restart_requested_B1_ignored", "restart_requested_B2_ignored")
             if c in df.columns]
    if not flags:
        return {}
    result = {}
    for anchor, group in df.groupby("anchor_frame"):
        steps = []
        for _, row in group.iterrows():
            if any(truthy(row[c]) for c in flags):
                steps.append(int(row["future_step"]))
        result[int(anchor)] = sorted(set(steps))
    return result


def valid_discount_weight(input_dir, sequence, anchor, horizon, gamma):
    """Sum gamma**k only for paired, finite AbsRel measurements at future step k."""
    path = input_dir / f"{sequence}_multihorizon_frame_metrics.csv"
    df = read_csv(path, required=False)
    if df.empty:
        return float("nan"), 0
    df = df[(df["anchor_frame"].astype(int) == int(anchor)) &
            (df["future_step"].astype(int) <= int(horizon))]
    if df.empty:
        return float("nan"), 0
    a = pd.to_numeric(df["absrel_B1_writeback"], errors="coerce")
    b = pd.to_numeric(df["absrel_B2_no_writeback"], errors="coerce")
    finite = np.isfinite(a.to_numpy(dtype=float)) & np.isfinite(b.to_numpy(dtype=float))
    steps = df.loc[finite, "future_step"].astype(int).to_numpy()
    return float(sum(gamma ** int(k) for k in steps)), int(len(steps))


def fit_predict_ridge(train_x, train_y, test_x, alpha):
    """Impute/scale/constant-column removal fitted strictly on training rows."""
    train_x = train_x.astype(float).copy()
    test_x = test_x.astype(float).copy()
    medians = train_x.median(axis=0).fillna(0.0)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)
    varying = train_x.nunique(dropna=False) > 1
    train_x = train_x.loc[:, varying]
    test_x = test_x.loc[:, varying]
    if train_x.shape[1] == 0:
        return np.full(len(test_x), float(np.mean(train_y)))
    scaler = StandardScaler()
    xtr = scaler.fit_transform(train_x)
    xte = scaler.transform(test_x)
    model = Ridge(alpha=alpha)
    model.fit(xtr, train_y)
    return model.predict(xte)


def spearman_safe(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(spearmanr(y, p).statistic)


def evaluate(y, pred, sequence, horizon, model_name, split):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    finite = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[finite], pred[finite]
    if not len(y):
        return {"split": split, "sequence": sequence, "horizon_H": horizon,
                "model": model_name, "n": 0}
    # Fixed threshold zero: no threshold tuning on test labels.
    actual_pos, predicted_pos = y > 0, pred > 0
    tp = int(np.sum(actual_pos & predicted_pos))
    tn = int(np.sum(~actual_pos & ~predicted_pos))
    fp = int(np.sum(~actual_pos & predicted_pos))
    fn = int(np.sum(actual_pos & ~predicted_pos))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    balanced = np.nanmean([sensitivity, specificity])
    try:
        auc = float(roc_auc_score(actual_pos, pred)) if len(np.unique(actual_pos)) == 2 else float("nan")
    except Exception:
        auc = float("nan")
    oracle = np.maximum(y, 0.0)
    chosen = np.where(predicted_pos, y, 0.0)
    return {
        "split": split, "sequence": sequence, "horizon_H": int(horizon),
        "model": model_name, "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "spearman": spearman_safe(y, pred),
        "sign_accuracy": float(np.mean(actual_pos == predicted_pos)),
        "balanced_accuracy": float(balanced),
        "positive_recall": float(sensitivity),
        "negative_recall": float(specificity),
        "negative_precision": float(tn / (tn + fn)) if tn + fn else float("nan"),
        "roc_auc": auc,
        "predicted_positive_fraction": float(np.mean(predicted_pos)),
        "true_positive_fraction": float(np.mean(actual_pos)),
        # Diagnostic binary writeback/no-writeback decision, no compute cost.
        "decision_regret_mean": float(np.mean(oracle - chosen)),
        "decision_reward_mean": float(np.mean(chosen)),
        "oracle_reward_mean": float(np.mean(oracle)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="outputs/p2_2_review")
    ap.add_argument("--out-dir", default="outputs/p2_2_1")
    ap.add_argument("--horizons", default="1,4,8,16")
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--target", default=TARGET)
    args = ap.parse_args()

    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted(set(int(x.strip()) for x in args.horizons.split(",") if x.strip()))
    if not horizons or any(h not in ALL_HORIZONS for h in horizons):
        ap.error("--horizons must be selected from 1,4,8,16")
    if not 0 <= args.min_coverage <= 1:
        ap.error("--min-coverage must be in [0,1]")

    candidates, eligible = [], []
    for seq in SEQUENCES:
        fpath = input_dir / f"pre_refresh_features_{seq}.csv"
        lpath = input_dir / f"{seq}_multihorizon_summary.csv"
        features = read_csv(fpath)
        labels = read_csv(lpath)
        required = {"anchor_frame", "horizon_H", args.target,
                    "future_metric_pair_coverage_ratio", "gamma"}
        missing = required - set(labels.columns)
        if missing:
            raise ValueError(f"{lpath} missing columns: {sorted(missing)}")
        if features["anchor_frame"].duplicated().any():
            raise ValueError(f"Duplicate feature anchor in {fpath}")
        features["anchor_frame"] = features["anchor_frame"].astype(int)
        labels["anchor_frame"] = labels["anchor_frame"].astype(int)
        labels["horizon_H"] = labels["horizon_H"].astype(int)
        labels = labels[labels["horizon_H"].isin(horizons)]
        restart_map = load_restart_steps(input_dir, seq)
        label_lookup = {(int(r.anchor_frame), int(r.horizon_H)): r
                        for r in labels.itertuples(index=False)}
        numeric_features = [
            c for c in features.columns
            if c not in {
                "sequence", "anchor_frame", "anchor_rgb_timestamp",
                "feature_timing", "label_source", "uses_future_gt_as_input",
                "uses_refresh_output_as_input", "scheduler_current_t",
                "scheduler_first_keyframe",
            } and pd.api.types.is_numeric_dtype(features[c])
        ]
        for _, feat in features.iterrows():
            anchor = int(feat["anchor_frame"])
            for h in horizons:
                lab = label_lookup.get((anchor, h))
                reasons = []
                rec = {
                    "sequence": seq, "anchor_frame": anchor, "horizon_H": h,
                    "target_name": args.target,
                }
                for c in numeric_features:
                    rec[c] = as_float(feat[c])
                if lab is None:
                    reasons.append("no_P1.5_label_row_anchor_skipped_or_missing")
                else:
                    y = as_float(getattr(lab, args.target))
                    coverage = as_float(lab.future_metric_pair_coverage_ratio)
                    rec[args.target] = y
                    for c in ("G_state_absrel_discounted", "G_state_absrel_discounted_per_step",
                              "G_state_rmse_discounted", "G_state_delta1_discounted",
                              "future_frames_requested", "future_frames_with_gt",
                              "future_gt_coverage_ratio", "future_frames_evaluated",
                              "future_metric_pair_coverage_ratio", "anchor_gt_available",
                              "gamma", "future_policy", "discount_policy"):
                        if hasattr(lab, c):
                            rec[c] = getattr(lab, c)
                    if not np.isfinite(y):
                        reasons.append("target_nonfinite")
                    if not np.isfinite(coverage) or coverage < args.min_coverage:
                        reasons.append("metric_pair_coverage_below_threshold")
                    # Anchor GT is recorded but not used as a hard exclusion.
                    rec["anchor_GT_available_audit_only"] = truthy(getattr(lab, "anchor_gt_available", False))
                    gamma = as_float(getattr(lab, "gamma", 0.9))
                    weight, finite_pairs = valid_discount_weight(input_dir, seq, anchor, h, gamma)
                    rec["valid_discount_weight_sum"] = weight
                    rec["finite_future_metric_pairs_recounted"] = finite_pairs
                    rec["target_discount_weight_normalized"] = (
                        as_float(getattr(lab, "G_state_absrel_discounted", np.nan)) / weight
                        if np.isfinite(weight) and weight > 0 else float("nan")
                    )
                    rec["target_discount_weight_normalized_valid"] = bool(
                        np.isfinite(rec["target_discount_weight_normalized"])
                    )
                crossed = [s for s in restart_map.get(anchor, []) if s <= h]
                rec["restart_steps_ignored_within_horizon"] = ",".join(map(str, crossed))
                if crossed:
                    reasons.append("future_window_crosses_ignored_restart")
                if any(not np.isfinite(as_float(rec.get(c))) for c in numeric_features):
                    reasons.append("nonfinite_feature_value")
                rec["eligible"] = not reasons
                rec["exclusion_reason"] = ";".join(dict.fromkeys(reasons))
                candidates.append(rec)
                if rec["eligible"]:
                    eligible.append(rec)

    audit = pd.DataFrame(candidates)
    valid = pd.DataFrame(eligible)
    audit.to_csv(out_dir / "p2_2_1_sample_audit.csv", index=False, encoding="utf-8-sig")
    valid.to_csv(out_dir / "p2_2_1_valid_dataset.csv", index=False, encoding="utf-8-sig")
    if valid.empty:
        raise RuntimeError("No eligible rows after audit; inspect p2_2_1_sample_audit.csv")

    metric_rows, pred_rows = [], []
    target_variants = [
        (args.target, args.target),
        ("target_discount_weight_normalized", "target_discount_weight_normalized"),
    ]
    for target_col, target_label in target_variants:
        if target_col not in valid.columns:
            continue
        for h in horizons:
            hdf = valid[valid["horizon_H"] == h].copy()
            hdf[target_col] = pd.to_numeric(hdf[target_col], errors="coerce")
            hdf = hdf[np.isfinite(hdf[target_col].to_numpy(dtype=float))]
            for held_seq in SEQUENCES:
                train = hdf[hdf["sequence"] != held_seq].copy()
                test = hdf[hdf["sequence"] == held_seq].copy()
                if len(train) < 8 or test.empty:
                    continue
                ytr = train[target_col].to_numpy(float)
                yte = test[target_col].to_numpy(float)
                model_predictions = {"mean_baseline": np.full(len(test), ytr.mean())}
                for name, cols in MODEL_FEATURES.items():
                    if name == "mean_baseline":
                        continue
                    missing = [c for c in cols if c not in hdf.columns]
                    if missing:
                        raise ValueError(f"Model {name} missing feature columns: {missing}")
                    model_predictions[name] = fit_predict_ridge(
                        train[cols], ytr, test[cols], args.ridge_alpha
                    )
                for model_name, pred in model_predictions.items():
                    metric = evaluate(yte, pred, held_seq, h, model_name, "held_out_sequence")
                    metric["target_variant"] = target_label
                    metric_rows.append(metric)
                    for j, (_, row) in enumerate(test.iterrows()):
                        pred_rows.append({
                            "sequence": held_seq,
                            "anchor_frame": int(row["anchor_frame"]),
                            "horizon_H": h,
                            "target_variant": target_label,
                            "model": model_name,
                            "y_true": float(yte[j]),
                            "y_pred": float(pred[j]),
                            "abs_error": float(abs(yte[j] - pred[j])),
                            "predicted_positive": bool(pred[j] > 0),
                            "true_positive": bool(yte[j] > 0),
                        })
            # Pooled metrics from out-of-sequence predictions only.
            pp = [r for r in pred_rows if r["horizon_H"] == h and r["target_variant"] == target_label]
            for model_name in MODEL_FEATURES:
                subset = [r for r in pp if r["model"] == model_name]
                if subset:
                    metric = evaluate(
                        [r["y_true"] for r in subset], [r["y_pred"] for r in subset],
                        "ALL", h, model_name, "pooled_LOSO"
                    )
                    metric["target_variant"] = target_label
                    metric_rows.append(metric)

    pred_df = pd.DataFrame(pred_rows)
    metrics_df = pd.DataFrame(metric_rows)
    pred_df.to_csv(out_dir / "p2_2_1_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_dir / "p2_2_1_metrics.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# SVR-Depth P2.2.1 Report", "",
        f"- Input: `{input_dir}`",
        f"- Primary target: `{args.target}`",
        f"- Minimum future metric-pair coverage: {args.min_coverage:.2f}",
        f"- Ridge alpha: {args.ridge_alpha:g}",
        "- Validation: leave one whole sequence out; all imputation, scaling and constant-column removal fit on training sequences only.",
        "- Anchor GT availability is audit-only, not an automatic exclusion.",
        "- Any horizon crossing a logged ignored restart is excluded.",
        "- Decision metrics use a fixed zero threshold; no test-fold threshold tuning.",
        "- Regret/reward is diagnostic binary refresh/no-refresh only and ignores compute cost.",
        "",
        "## Dataset audit", "",
        f"- Candidate rows: {len(audit)}",
        f"- Eligible rows: {len(valid)}",
        f"- Excluded rows: {len(audit)-len(valid)}",
        "",
        "| Sequence | H | Candidate | Eligible |", "|---|---:|---:|---:|",
    ]
    for seq in SEQUENCES:
        for h in horizons:
            sub = audit[(audit.sequence == seq) & (audit.horizon_H == h)]
            nvalid = int(sub.eligible.sum())
            lines.append(f"| {seq} | {h} | {len(sub)} | {nvalid} |")
    lines += ["", "### Exclusion reasons", "", "| Reason | Count |", "|---|---:|"]
    reason_counts = {}
    for s in audit.exclusion_reason.fillna(""):
        for reason in filter(None, s.split(";")):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {reason} | {count} |")
    lines += [
        "", "## Pooled leave-one-sequence-out results", "",
        "| Target | H | Model | N | MAE | RMSE | Spearman | Balanced acc. | Pos recall | Neg recall | AUC | Regret |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pooled = metrics_df[metrics_df.split == "pooled_LOSO"] if not metrics_df.empty else pd.DataFrame()
    for _, r in pooled.iterrows():
        def fmt(v):
            return "NA" if pd.isna(v) else f"{v:.4g}"
        lines.append(
            f"| {r.target_variant} | {int(r.horizon_H)} | {r.model} | {int(r.n)} | "
            f"{fmt(r.get('mae', np.nan))} | {fmt(r.get('rmse', np.nan))} | "
            f"{fmt(r.get('spearman', np.nan))} | {fmt(r.get('balanced_accuracy', np.nan))} | "
            f"{fmt(r.get('positive_recall', np.nan))} | {fmt(r.get('negative_recall', np.nan))} | "
            f"{fmt(r.get('roc_auc', np.nan))} | {fmt(r.get('decision_regret_mean', np.nan))} |"
        )
    lines += [
        "", "## Interpretation rules", "",
        "1. Compare `geometry_only`, `confidence_only`, `geometry_confidence`, `learned_state_only`, and `state_motion_full` to isolate which state family carries signal.",
        "2. Compare balanced accuracy, negative recall, AUC and regret—not ordinary accuracy alone.",
        "3. Per-sequence rows are primary for judging transfer. Pooled rows can hide scene-specific failures.",
        "4. The normalized-discount target is a sensitivity analysis, not a replacement for the original target.",
        "5. These are only three held-out sequences and overlapping temporal windows; treat as exploratory evidence.",
        "6. Target describes full-frame state-writeback under forced propagation, not real online B0/B1/B2 or local refresh.",
        "",
        "## Outputs", "",
        "- `p2_2_1_sample_audit.csv`",
        "- `p2_2_1_valid_dataset.csv`",
        "- `p2_2_1_predictions.csv`",
        "- `p2_2_1_metrics.csv`",
        "- `p2_2_1_report.md`",
    ]
    (out_dir / "p2_2_1_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("=== P2.2.1 complete ===")
    print(f"Candidate rows: {len(audit)}; eligible rows: {len(valid)}; excluded: {len(audit)-len(valid)}")
    print("Output:", out_dir)
    print("\nPooled LOSO metrics:")
    if not pooled.empty:
        cols = [c for c in ["target_variant", "horizon_H", "model", "n", "mae",
                            "spearman", "balanced_accuracy", "negative_recall",
                            "roc_auc", "decision_regret_mean"] if c in pooled.columns]
        print(pooled[cols].to_string(index=False))


if __name__ == "__main__":
    main()
