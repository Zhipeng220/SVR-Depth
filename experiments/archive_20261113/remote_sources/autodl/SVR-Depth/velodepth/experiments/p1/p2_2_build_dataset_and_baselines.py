#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVR-Depth P2.2: build a leakage-safe feature/label dataset and evaluate simple
leave-one-sequence-out baselines.

Expected input directory (default: outputs/p2_prediction_review):
  pre_refresh_features_crowd2.csv
  pre_refresh_features_balloon2.csv
  pre_refresh_features_crowd3.csv
  crowd2_multihorizon_summary.csv
  balloon2_multihorizon_summary.csv
  crowd3_multihorizon_summary.csv
  crowd2_multihorizon_frame_metrics.csv
  balloon2_multihorizon_frame_metrics.csv
  crowd3_multihorizon_frame_metrics.csv

Outputs (default: outputs/p2_prediction):
  p2_2_sample_audit.csv       all feature x horizon rows, including exclusions
  p2_2_valid_dataset.csv      rows eligible for modeling
  p2_2_predictions.csv        leave-one-sequence-out predictions
  p2_2_metrics.csv            fold and pooled metrics
  p2_2_report.md              human-readable run report

Important interpretation:
- The target is the P1.5 discounted future AbsRel gain per evaluated valid frame.
  Positive means state writeback improves future AbsRel under the P1.5 forced-
  propagation counterfactual, not necessarily under a real online scheduler.
- Samples whose P1.5 future window contains an ignored restart request are
  excluded for that horizon.
- No sequence ID, anchor index, timestamp, current_t, GT, or refresh output is
  used as a predictor. The "state+motion" model tests whether compact persistent
  geometry/confidence summaries add value beyond age/motion observations.
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

try:
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:
    raise SystemExit(
        "P2.2 requires scikit-learn. Install with: pip install scikit-learn scipy"
    ) from exc


SEQUENCES = ("crowd2", "balloon2", "crowd3")
HORIZONS = (1, 4, 8, 16)
TARGET = "G_state_absrel_discounted_per_valid_frame"
MIN_COVERAGE = 0.80
RIDGE_ALPHA = 10.0

AGE_MOTION_FEATURES = [
    "scheduler_distance_kf",
    "scheduler_flow_magnitude",
    "scheduler_total_luma_change",
    "image_absdiff_mean_prev",
    "image_absdiff_p90_prev",
]

STATE_FEATURES = AGE_MOTION_FEATURES + [
    "state_radius_teacher_mean",
    "state_radius_teacher_std",
    "state_radius_teacher_q10",
    "state_radius_teacher_q50",
    "state_radius_teacher_q90",
    "state_confidence_teacher_mean",
    "state_confidence_teacher_std",
    "state_confidence_teacher_q10",
    "state_confidence_teacher_q50",
    "state_confidence_teacher_q90",
    "state_features_teacher_0_mean",
    "state_features_teacher_0_std",
    "state_features_teacher_1_mean",
    "state_features_teacher_1_std",
    "state_features_teacher_2_mean",
    "state_features_teacher_2_std",
    "state_features_teacher_3_mean",
    "state_features_teacher_3_std",
]

MODEL_FEATURES = {
    "mean_baseline": [],
    "age_motion_ridge": AGE_MOTION_FEATURES,
    "state_motion_ridge": STATE_FEATURES,
}


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_csv(path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def spearman_value(y_true, y_pred):
    if len(y_true) < 3 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return float("nan")
    if spearmanr is None:
        return float("nan")
    result = spearmanr(y_true, y_pred, nan_policy="omit")
    return float(result.statistic) if hasattr(result, "statistic") else float(result[0])


def metric_row(y_true, y_pred, sequence, horizon, model, split):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) == 0:
        return {
            "split": split, "sequence": sequence, "horizon_H": horizon,
            "model": model, "n": 0, "mae": np.nan, "rmse": np.nan,
            "spearman": np.nan, "sign_accuracy": np.nan,
            "positive_fraction": np.nan, "diagnostic_sign_regret": np.nan,
        }
    # Sign accuracy is defined with exact zero treated as non-positive.
    sign_acc = float(np.mean((p > 0) == (y > 0)))
    # Diagnostic only: assumes a binary writeback/no-writeback choice and no
    # compute cost. It is not the final B0/B1/B2 policy utility.
    oracle_reward = np.maximum(y, 0.0)
    chosen_reward = np.where(p > 0, y, 0.0)
    regret = float(np.mean(oracle_reward - chosen_reward))
    return {
        "split": split,
        "sequence": sequence,
        "horizon_H": int(horizon),
        "model": model,
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "spearman": spearman_value(y, p),
        "sign_accuracy": sign_acc,
        "positive_fraction": float(np.mean(y > 0)),
        "diagnostic_sign_regret": regret,
    }


def make_pipeline(alpha):
    # All transformations are fit inside each training fold to avoid leakage.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def read_restart_map(input_dir, sequence):
    path = input_dir / f"{sequence}_multihorizon_frame_metrics.csv"
    frame = load_csv(path, required=False)
    if frame.empty:
        return {}
    needed = {"anchor_frame", "future_step"}
    if not needed.issubset(frame.columns):
        return {}
    flag_cols = [c for c in (
        "restart_requested_B1_ignored", "restart_requested_B2_ignored"
    ) if c in frame.columns]
    if not flag_cols:
        return {}
    result = {}
    for anchor, group in frame.groupby("anchor_frame"):
        steps = []
        for _, row in group.iterrows():
            if any(parse_bool(row[c]) for c in flag_cols):
                steps.append(int(row["future_step"]))
        result[int(anchor)] = sorted(set(steps))
    return result


def main():
    parser = argparse.ArgumentParser(description="SVR-Depth P2.2 dataset + baselines")
    parser.add_argument("--input-dir", default="outputs/p2_prediction_review")
    parser.add_argument("--out-dir", default="outputs/p2_prediction")
    parser.add_argument("--horizons", default="1,4,8,16")
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--target", default=TARGET)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted(set(int(x.strip()) for x in args.horizons.split(",") if x.strip()))
    if not horizons or any(h not in HORIZONS for h in horizons):
        parser.error("horizons must be selected from 1,4,8,16")
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage must be in [0,1]")

    audit_rows = []
    valid_rows = []
    feature_columns_by_sequence = {}
    required_target_cols = {
        "anchor_frame", "horizon_H", args.target,
        "future_metric_pair_coverage_ratio", "anchor_gt_available",
    }

    for sequence in SEQUENCES:
        feature_path = input_dir / f"pre_refresh_features_{sequence}.csv"
        summary_path = input_dir / f"{sequence}_multihorizon_summary.csv"
        features = load_csv(feature_path)
        labels = load_csv(summary_path)
        missing = required_target_cols - set(labels.columns)
        if missing:
            raise ValueError(f"{summary_path} missing columns: {sorted(missing)}")
        if "anchor_frame" not in features.columns:
            raise ValueError(f"{feature_path} missing anchor_frame")
        if features["anchor_frame"].duplicated().any():
            raise ValueError(f"Duplicate anchor_frame rows in {feature_path}")

        features["anchor_frame"] = features["anchor_frame"].astype(int)
        labels["anchor_frame"] = labels["anchor_frame"].astype(int)
        labels["horizon_H"] = labels["horizon_H"].astype(int)
        labels = labels[labels["horizon_H"].isin(horizons)].copy()
        merged = features.merge(
            labels, on="anchor_frame", how="left", suffixes=("", "_label"),
            indicator=True,
        )
        # Feature columns are numeric pre-refresh measurements only. Metadata,
        # timestamps, IDs, flags, and every label-derived field are excluded.
        feature_cols = [
            c for c in features.columns
            if c not in {
                "sequence", "anchor_frame", "anchor_rgb_timestamp",
                "feature_timing", "label_source", "uses_future_gt_as_input",
                "uses_refresh_output_as_input", "scheduler_current_t",
                "scheduler_first_keyframe",
            }
            and pd.api.types.is_numeric_dtype(features[c])
        ]
        feature_columns_by_sequence[sequence] = feature_cols
        restart_map = read_restart_map(input_dir, sequence)

        for _, row in merged.iterrows():
            # A feature row can join to several horizon labels, so process each
            # horizon explicitly and preserve a row even when the label is absent.
            candidate_horizons = horizons if row["_merge"] == "left_only" else [int(row["horizon_H"])]
            for horizon in candidate_horizons:
                out = {
                    "sequence": sequence,
                    "anchor_frame": int(row["anchor_frame"]),
                    "horizon_H": int(horizon),
                    "target_name": args.target,
                }
                for c in feature_cols:
                    out[c] = as_float(row[c])
                reasons = []
                if row["_merge"] == "left_only":
                    reasons.append("no_P1.5_label_row_anchor_skipped_or_missing")
                    label_row = None
                else:
                    # Because merge can create multiple horizon rows, select the
                    # matching label record from the original label table.
                    match = labels[
                        (labels["anchor_frame"] == int(row["anchor_frame"])) &
                        (labels["horizon_H"] == int(horizon))
                    ]
                    label_row = match.iloc[0] if len(match) else None
                    if label_row is None:
                        reasons.append("missing_horizon_label")

                if label_row is not None:
                    for col in (
                        args.target, "G_state_absrel_discounted",
                        "G_state_absrel_discounted_per_step",
                        "G_state_rmse_discounted", "G_state_delta1_discounted",
                        "future_frames_requested", "future_frames_with_gt",
                        "future_gt_coverage_ratio", "future_frames_evaluated",
                        "future_metric_pair_coverage_ratio", "anchor_gt_available",
                        "future_policy", "discount_policy",
                    ):
                        if col in label_row.index:
                            out[col] = label_row[col]
                    y = as_float(label_row[args.target])
                    coverage = as_float(label_row["future_metric_pair_coverage_ratio"])
                    if not parse_bool(label_row["anchor_gt_available"]):
                        reasons.append("anchor_GT_unavailable")
                    if not math.isfinite(y):
                        reasons.append("target_nonfinite")
                    if not math.isfinite(coverage) or coverage < args.min_coverage:
                        reasons.append("metric_pair_coverage_below_threshold")
                else:
                    y = float("nan")
                    coverage = float("nan")

                restart_steps = restart_map.get(int(row["anchor_frame"]), [])
                crossed = [s for s in restart_steps if s <= horizon]
                out["restart_steps_ignored_within_horizon"] = ",".join(map(str, crossed))
                if crossed:
                    reasons.append("future_window_crosses_ignored_restart")

                # Exclude invalid predictor rows, but record the reason.
                if any(not math.isfinite(as_float(out.get(c))) for c in feature_cols):
                    reasons.append("nonfinite_feature_value")
                out["eligible"] = not reasons
                out["exclusion_reason"] = ";".join(dict.fromkeys(reasons))
                audit_rows.append(out)
                if out["eligible"]:
                    valid_rows.append(out)

    audit = pd.DataFrame(audit_rows)
    valid = pd.DataFrame(valid_rows)
    if audit.empty:
        raise RuntimeError("No joined feature/label rows found. Check --input-dir.")
    audit.to_csv(out_dir / "p2_2_sample_audit.csv", index=False, encoding="utf-8-sig")
    valid.to_csv(out_dir / "p2_2_valid_dataset.csv", index=False, encoding="utf-8-sig")

    if valid.empty:
        raise RuntimeError("No eligible rows remain after label/coverage/restart checks.")
    metrics = []
    predictions = []
    for horizon in horizons:
        hdf = valid[valid["horizon_H"] == horizon].copy()
        if hdf.empty:
            continue
        y_col = args.target
        # Convert the chosen target explicitly; do not let pandas coerce strings.
        hdf[y_col] = pd.to_numeric(hdf[y_col], errors="coerce")
        hdf = hdf[np.isfinite(hdf[y_col].to_numpy(dtype=float))]
        if hdf.empty:
            continue

        for held_sequence in SEQUENCES:
            train = hdf[hdf["sequence"] != held_sequence].copy()
            test = hdf[hdf["sequence"] == held_sequence].copy()
            if len(train) < 8 or test.empty:
                continue
            y_train = train[y_col].to_numpy(dtype=float)
            y_test = test[y_col].to_numpy(dtype=float)
            mean_pred = np.full(len(test), float(np.mean(y_train)))
            fold_preds = {
                "mean_baseline": mean_pred,
            }
            for model_name in ("age_motion_ridge", "state_motion_ridge"):
                columns = MODEL_FEATURES[model_name]
                missing_cols = [c for c in columns if c not in hdf.columns]
                if missing_cols:
                    raise ValueError(
                        f"Missing expected feature columns for {model_name}: {missing_cols}"
                    )
                model = make_pipeline(args.ridge_alpha)
                model.fit(train[columns].astype(float), y_train)
                fold_preds[model_name] = model.predict(test[columns].astype(float))

            for model_name, pred in fold_preds.items():
                metrics.append(metric_row(
                    y_test, pred, held_sequence, horizon, model_name, "held_out_sequence"
                ))
                for i, (_, sample) in enumerate(test.iterrows()):
                    predictions.append({
                        "sequence": sample["sequence"],
                        "anchor_frame": int(sample["anchor_frame"]),
                        "horizon_H": int(horizon),
                        "model": model_name,
                        "target_name": y_col,
                        "y_true": float(y_test[i]),
                        "y_pred": float(pred[i]),
                        "abs_error": float(abs(y_test[i] - pred[i])),
                        "sign_correct": bool((pred[i] > 0) == (y_test[i] > 0)),
                        "held_out_sequence": held_sequence,
                    })

        # Pooled metrics use only out-of-sequence predictions.
        pred_h = [p for p in predictions if p["horizon_H"] == horizon]
        for model_name in MODEL_FEATURES:
            subset = [p for p in pred_h if p["model"] == model_name]
            if subset:
                metrics.append(metric_row(
                    [p["y_true"] for p in subset],
                    [p["y_pred"] for p in subset],
                    "ALL", horizon, model_name, "pooled_LOSO",
                ))

    pred_df = pd.DataFrame(predictions)
    metrics_df = pd.DataFrame(metrics)
    pred_df.to_csv(out_dir / "p2_2_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_dir / "p2_2_metrics.csv", index=False, encoding="utf-8-sig")

    # A compact report with sample counts and pooled held-out metrics.
    lines = [
        "# SVR-Depth P2.2 Report",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Target: `{args.target}`",
        f"- Minimum metric-pair coverage: {args.min_coverage:.2f}",
        f"- Ridge alpha: {args.ridge_alpha:g}",
        "- Validation: leave one entire sequence out; all preprocessing is fitted on training sequences only.",
        "- Positive target means writeback reduces future AbsRel in the P1.5 forced-propagation counterfactual.",
        "- Binary sign regret is diagnostic only and ignores compute cost; it is not the final online policy utility.",
        "",
        "## Dataset audit",
        "",
        f"- Feature/label candidate rows: {len(audit)}",
        f"- Eligible rows: {len(valid)}",
        f"- Excluded rows: {len(audit) - len(valid)}",
        "",
        "### Eligible samples by sequence and horizon",
        "",
        "| Sequence | H | Eligible samples |",
        "|---|---:|---:|",
    ]
    for sequence in SEQUENCES:
        for horizon in horizons:
            n = int(((valid["sequence"] == sequence) & (valid["horizon_H"] == horizon)).sum())
            lines.append(f"| {sequence} | {horizon} | {n} |")
    lines += [
        "",
        "### Exclusion reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    reason_counts = {}
    for value in audit["exclusion_reason"].fillna(""):
        for reason in filter(None, value.split(";")):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| None | 0 |")
    lines += [
        "",
        "### Pooled leave-one-sequence-out metrics",
        "",
        "| H | Model | N | MAE | RMSE | Spearman | Sign accuracy | Diagnostic sign regret |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not metrics_df.empty:
        pooled = metrics_df[metrics_df["split"] == "pooled_LOSO"]
        for _, r in pooled.sort_values(["horizon_H", "model"]).iterrows():
            lines.append(
                f"| {int(r['horizon_H'])} | {r['model']} | {int(r['n'])} | "
                f"{r['mae']:.6g} | {r['rmse']:.6g} | {r['spearman']:.4g} | "
                f"{r['sign_accuracy']:.4g} | {r['diagnostic_sign_regret']:.6g} |"
            )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "1. Treat this as a small-sample feasibility test, not a definitive generalization claim.",
        "2. Compare `state_motion_ridge` against `age_motion_ridge`; improvement is evidence that state summaries add signal beyond the compact age/motion baseline.",
        "3. Inspect each held-out sequence fold. Pooled metrics alone can hide a failure on one scene.",
        "4. H=1,4,8,16 are related targets from overlapping windows; they are not independent replications.",
        "5. The current target describes a full-frame refresh-state-writeback oracle under forced propagation. It does not yet establish the value of B0/B1/B2 online actions or local refresh.",
        "",
        "## Output files",
        "",
        "- `p2_2_sample_audit.csv`: all candidates and exclusion reasons.",
        "- `p2_2_valid_dataset.csv`: modeling-eligible rows.",
        "- `p2_2_predictions.csv`: held-out predictions per sample and model.",
        "- `p2_2_metrics.csv`: fold-level and pooled metrics.",
        "- `p2_2_report.md`: this report.",
        "",
    ]
    (out_dir / "p2_2_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("=== SVR-Depth P2.2 complete ===")
    print("Candidate rows:", len(audit))
    print("Eligible rows:", len(valid))
    print("Excluded rows:", len(audit) - len(valid))
    print("Output directory:", out_dir)
    print(" - p2_2_sample_audit.csv")
    print(" - p2_2_valid_dataset.csv")
    print(" - p2_2_predictions.csv")
    print(" - p2_2_metrics.csv")
    print(" - p2_2_report.md")
    print("\nPooled LOSO metrics:")
    if not metrics_df.empty:
        cols = ["horizon_H", "model", "n", "mae", "rmse", "spearman", "sign_accuracy"]
        print(metrics_df[metrics_df["split"] == "pooled_LOSO"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
