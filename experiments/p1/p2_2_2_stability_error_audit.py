#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVR-Depth P2.2.2: per-sequence stability, incremental feature value, and
decision-error audit.

Run from the VeloDepth repository:
python experiments/p1/p2_2_2_stability_error_audit.py \
  --input-dir outputs/p2_2_review \
  --out-dir outputs/p2_2_2 \
  --horizons 1,4,8,16

CPU only. No VeloDepth inference or GPU required.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEQUENCES = ("crowd2", "balloon2", "crowd3")
ALLOWED_HORIZONS = (1, 4, 8, 16)
DEFAULT_TARGET = "G_state_absrel_discounted_per_valid_frame"

MOTION = [
    "scheduler_distance_kf",
    "scheduler_flow_magnitude",
    "scheduler_total_luma_change",
    "image_absdiff_mean_prev",
    "image_absdiff_p90_prev",
]
GEOMETRY = [
    "state_radius_teacher_mean", "state_radius_teacher_std",
    "state_radius_teacher_q10", "state_radius_teacher_q50",
    "state_radius_teacher_q90", "state_rays_base_mean",
    "state_rays_base_std", "state_rays_base_q10",
    "state_rays_base_q50", "state_rays_base_q90",
]
CONFIDENCE = [
    "state_confidence_teacher_mean", "state_confidence_teacher_std",
    "state_confidence_teacher_q10", "state_confidence_teacher_q50",
    "state_confidence_teacher_q90",
]
LEARNED = [
    f"state_features_teacher_{i}_{stat}"
    for i in range(4) for stat in ("mean", "std")
]

# Nested/near-nested ablations to estimate whether one feature family adds
# information beyond another under the same LOSO protocol.
MODEL_FEATURES = {
    "mean_baseline": [],
    "motion_only": MOTION,
    "learned_state_only": LEARNED,
    "motion_plus_learned": MOTION + LEARNED,
    "geometry_confidence_only": GEOMETRY + CONFIDENCE,
    "motion_plus_geom_conf": MOTION + GEOMETRY + CONFIDENCE,
    "state_motion_full": MOTION + GEOMETRY + CONFIDENCE + LEARNED,
}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def boolish(x):
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


def restart_steps_by_anchor(input_dir, seq):
    path = input_dir / f"{seq}_multihorizon_frame_metrics.csv"
    df = read_csv(path, required=False)
    if df.empty or not {"anchor_frame", "future_step"}.issubset(df.columns):
        return {}
    flags = [c for c in ("restart_requested_B1_ignored", "restart_requested_B2_ignored")
             if c in df.columns]
    if not flags:
        return {}
    out = {}
    for anchor, group in df.groupby("anchor_frame"):
        steps = []
        for _, row in group.iterrows():
            if any(boolish(row[c]) for c in flags):
                steps.append(int(row["future_step"]))
        out[int(anchor)] = sorted(set(steps))
    return out


def discount_weight(input_dir, seq, anchor, horizon, gamma):
    path = input_dir / f"{seq}_multihorizon_frame_metrics.csv"
    df = read_csv(path, required=False)
    if df.empty:
        return np.nan, 0
    df = df[(df.anchor_frame.astype(int) == int(anchor)) &
            (df.future_step.astype(int) <= int(horizon))]
    if df.empty:
        return np.nan, 0
    required = ["absrel_B1_writeback", "absrel_B2_no_writeback"]
    if not set(required).issubset(df.columns):
        return np.nan, 0
    a = pd.to_numeric(df[required[0]], errors="coerce").to_numpy(float)
    b = pd.to_numeric(df[required[1]], errors="coerce").to_numpy(float)
    good = np.isfinite(a) & np.isfinite(b)
    steps = df.loc[good, "future_step"].astype(int).to_numpy()
    return float(sum(gamma ** int(k) for k in steps)), int(len(steps))


def fit_predict(train_x, train_y, test_x, alpha):
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


def safe_spearman(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    # Exact uniqueness check avoids scipy ConstantInputWarning on near-constant
    # predictions whose ranks collapse to a single value.
    if len(y) < 3 or np.unique(y).size < 2 or np.unique(p).size < 2:
        return np.nan
    with np.errstate(all="ignore"):
        value = spearmanr(y, p).statistic
    return float(value) if np.isfinite(value) else np.nan


def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    good = np.isfinite(y) & np.isfinite(p)
    y, p = y[good], p[good]
    if not len(y):
        return {"n": 0}
    true_pos = y > 0
    pred_pos = p > 0
    tp = int(np.sum(true_pos & pred_pos))
    tn = int(np.sum(~true_pos & ~pred_pos))
    fp = int(np.sum(~true_pos & pred_pos))
    fn = int(np.sum(true_pos & ~pred_pos))
    pos_recall = tp / (tp + fn) if tp + fn else np.nan
    neg_recall = tn / (tn + fp) if tn + fp else np.nan
    try:
        auc = float(roc_auc_score(true_pos, p)) if len(np.unique(true_pos)) == 2 else np.nan
    except Exception:
        auc = np.nan
    reward_oracle = np.maximum(y, 0.0)
    reward_policy = np.where(pred_pos, y, 0.0)
    return {
        "n": len(y),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "spearman": safe_spearman(y, p),
        "sign_accuracy": float(np.mean(true_pos == pred_pos)),
        "balanced_accuracy": float(np.nanmean([pos_recall, neg_recall])),
        "positive_recall": pos_recall,
        "negative_recall": neg_recall,
        "roc_auc": auc,
        "predicted_positive_fraction": float(np.mean(pred_pos)),
        "true_positive_fraction": float(np.mean(true_pos)),
        "false_refresh_count": fp,
        "missed_benefit_count": fn,
        "decision_regret_mean": float(np.mean(reward_oracle - reward_policy)),
        "decision_reward_mean": float(np.mean(reward_policy)),
        "oracle_reward_mean": float(np.mean(reward_oracle)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="outputs/p2_2_review")
    ap.add_argument("--out-dir", default="outputs/p2_2_2")
    ap.add_argument("--horizons", default="1,4,8,16")
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    args = ap.parse_args()

    inp, outdir = Path(args.input_dir), Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    horizons = sorted(set(int(v.strip()) for v in args.horizons.split(",") if v.strip()))
    if not horizons or any(h not in ALLOWED_HORIZONS for h in horizons):
        ap.error("--horizons must be selected from 1,4,8,16")

    candidate_rows, valid_rows = [], []
    for seq in SEQUENCES:
        features = read_csv(inp / f"pre_refresh_features_{seq}.csv")
        labels = read_csv(inp / f"{seq}_multihorizon_summary.csv")
        needed = {"anchor_frame", "horizon_H", args.target, "future_metric_pair_coverage_ratio", "gamma"}
        missing = needed - set(labels.columns)
        if missing:
            raise ValueError(f"{seq} label CSV missing columns: {sorted(missing)}")
        features["anchor_frame"] = features.anchor_frame.astype(int)
        labels["anchor_frame"] = labels.anchor_frame.astype(int)
        labels["horizon_H"] = labels.horizon_H.astype(int)
        labels = labels[labels.horizon_H.isin(horizons)]
        lookup = {(int(r.anchor_frame), int(r.horizon_H)): r for r in labels.itertuples(index=False)}
        restarts = restart_steps_by_anchor(inp, seq)
        feature_cols = [
            c for c in features.columns
            if c not in {"sequence", "anchor_frame", "anchor_rgb_timestamp", "feature_timing",
                         "label_source", "uses_future_gt_as_input", "uses_refresh_output_as_input",
                         "scheduler_current_t", "scheduler_first_keyframe"}
            and pd.api.types.is_numeric_dtype(features[c])
        ]
        for _, fr in features.iterrows():
            anchor = int(fr.anchor_frame)
            for h in horizons:
                lab = lookup.get((anchor, h))
                row = {"sequence": seq, "anchor_frame": anchor, "horizon_H": h}
                for c in feature_cols:
                    row[c] = num(fr[c])
                reasons = []
                if lab is None:
                    reasons.append("missing_or_skipped_label")
                else:
                    y = num(getattr(lab, args.target))
                    coverage = num(lab.future_metric_pair_coverage_ratio)
                    row[args.target] = y
                    for c in ("G_state_absrel_discounted", "G_state_absrel_discounted_per_valid_frame",
                              "future_frames_requested", "future_frames_with_gt",
                              "future_gt_coverage_ratio", "future_frames_evaluated",
                              "future_metric_pair_coverage_ratio", "anchor_gt_available",
                              "gamma", "future_policy", "discount_policy"):
                        if hasattr(lab, c):
                            row[c] = getattr(lab, c)
                    row["anchor_GT_available_audit_only"] = boolish(getattr(lab, "anchor_gt_available", False))
                    if not np.isfinite(y):
                        reasons.append("target_nonfinite")
                    if not np.isfinite(coverage) or coverage < args.min_coverage:
                        reasons.append("future_pair_coverage_below_threshold")
                    gamma = num(getattr(lab, "gamma", 0.9))
                    wsum, npairs = discount_weight(inp, seq, anchor, h, gamma)
                    row["valid_discount_weight_sum"] = wsum
                    row["finite_future_metric_pairs_recounted"] = npairs
                    raw_discounted = num(getattr(lab, "G_state_absrel_discounted", np.nan))
                    row["target_discount_weight_normalized"] = (
                        raw_discounted / wsum if np.isfinite(wsum) and wsum > 0 else np.nan
                    )
                crossed = [step for step in restarts.get(anchor, []) if step <= h]
                row["restart_steps_ignored_within_horizon"] = ",".join(map(str, crossed))
                if crossed:
                    reasons.append("crosses_ignored_restart")
                if any(not np.isfinite(num(row.get(c))) for c in feature_cols):
                    reasons.append("nonfinite_feature")
                row["eligible"] = not reasons
                row["exclusion_reason"] = ";".join(dict.fromkeys(reasons))
                candidate_rows.append(row)
                if row["eligible"]:
                    valid_rows.append(row)

    audit = pd.DataFrame(candidate_rows)
    valid = pd.DataFrame(valid_rows)
    audit.to_csv(outdir / "p2_2_2_sample_audit.csv", index=False, encoding="utf-8-sig")
    valid.to_csv(outdir / "p2_2_2_valid_dataset.csv", index=False, encoding="utf-8-sig")
    if valid.empty:
        raise RuntimeError("No valid samples; inspect the sample audit.")

    metrics, predictions = [], []
    targets = [args.target, "target_discount_weight_normalized"]
    for target in targets:
        if target not in valid.columns:
            continue
        for h in horizons:
            dfh = valid[valid.horizon_H == h].copy()
            dfh[target] = pd.to_numeric(dfh[target], errors="coerce")
            dfh = dfh[np.isfinite(dfh[target].to_numpy(float))]
            for held in SEQUENCES:
                train = dfh[dfh.sequence != held]
                test = dfh[dfh.sequence == held]
                if len(train) < 8 or test.empty:
                    continue
                ytr = train[target].to_numpy(float)
                yte = test[target].to_numpy(float)
                pred_map = {"mean_baseline": np.full(len(test), ytr.mean())}
                for name, cols in MODEL_FEATURES.items():
                    if name == "mean_baseline":
                        continue
                    missing_cols = [c for c in cols if c not in dfh.columns]
                    if missing_cols:
                        raise ValueError(f"Model {name} missing columns: {missing_cols}")
                    pred_map[name] = fit_predict(train[cols], ytr, test[cols], args.ridge_alpha)
                for model, pred in pred_map.items():
                    met = score(yte, pred)
                    met.update({"split": "held_out_sequence", "sequence": held,
                                "horizon_H": h, "target_variant": target, "model": model})
                    metrics.append(met)
                    for idx, (_, sample) in enumerate(test.iterrows()):
                        y = float(yte[idx]); p = float(pred[idx])
                        predictions.append({
                            "sequence": held, "anchor_frame": int(sample.anchor_frame),
                            "horizon_H": h, "target_variant": target, "model": model,
                            "y_true": y, "y_pred": p, "abs_error": abs(y-p),
                            "true_positive": bool(y > 0), "predicted_positive": bool(p > 0),
                            "error_type": ("false_refresh" if y <= 0 and p > 0 else
                                           "missed_benefit" if y > 0 and p <= 0 else "correct"),
                            "anchor_GT_available_audit_only": sample.get("anchor_GT_available_audit_only", np.nan),
                            "restart_steps_ignored_within_horizon": sample.get("restart_steps_ignored_within_horizon", ""),
                        })
            # Pooled out-of-sequence rows
            pframe = pd.DataFrame([r for r in predictions
                                   if r["horizon_H"] == h and r["target_variant"] == target])
            for model in MODEL_FEATURES:
                sub = pframe[pframe.model == model]
                if sub.empty:
                    continue
                met = score(sub.y_true.to_numpy(float), sub.y_pred.to_numpy(float))
                met.update({"split": "pooled_LOSO", "sequence": "ALL",
                            "horizon_H": h, "target_variant": target, "model": model})
                metrics.append(met)

    pred_df = pd.DataFrame(predictions)
    metric_df = pd.DataFrame(metrics)
    pred_df.to_csv(outdir / "p2_2_2_predictions.csv", index=False, encoding="utf-8-sig")
    metric_df.to_csv(outdir / "p2_2_2_metrics.csv", index=False, encoding="utf-8-sig")

    # Incremental feature comparisons: differences are paired within same held-out
    # samples, so delta MAE < 0 favors the augmented model.
    pairs = [
        ("motion_only", "motion_plus_learned", "learned_state_increment_beyond_motion"),
        ("learned_state_only", "motion_plus_learned", "motion_increment_beyond_learned_state"),
        ("motion_plus_learned", "state_motion_full", "geometry_confidence_increment"),
        ("motion_only", "state_motion_full", "all_state_features_increment_beyond_motion"),
    ]
    increments = []
    for (target, h, seq), group in pred_df.groupby(["target_variant", "horizon_H", "sequence"]):
        wide = group.pivot_table(index="anchor_frame", columns="model",
                                 values=["y_true", "y_pred", "abs_error"], aggfunc="first")
        for base, augmented, label in pairs:
            if ("abs_error", base) not in wide or ("abs_error", augmented) not in wide:
                continue
            common = wide.dropna(subset=[("abs_error", base), ("abs_error", augmented)])
            if common.empty:
                continue
            mae_base = float(common[("abs_error", base)].mean())
            mae_aug = float(common[("abs_error", augmented)].mean())
            # Same target rows, so calculate ranking/decision deltas from predictions.
            y = common[("y_true", base)].to_numpy(float)
            p_base = common[("y_pred", base)].to_numpy(float)
            p_aug = common[("y_pred", augmented)].to_numpy(float)
            sbase, saug = score(y, p_base), score(y, p_aug)
            increments.append({
                "target_variant": target, "horizon_H": h, "sequence": seq,
                "comparison": label, "base_model": base, "augmented_model": augmented,
                "n_paired": len(common), "mae_base": mae_base, "mae_augmented": mae_aug,
                "delta_mae_aug_minus_base": mae_aug - mae_base,
                "delta_spearman_aug_minus_base": (
                    saug.get("spearman", np.nan) - sbase.get("spearman", np.nan)
                    if np.isfinite(saug.get("spearman", np.nan)) and np.isfinite(sbase.get("spearman", np.nan)) else np.nan
                ),
                "delta_auc_aug_minus_base": (
                    saug.get("roc_auc", np.nan) - sbase.get("roc_auc", np.nan)
                    if np.isfinite(saug.get("roc_auc", np.nan)) and np.isfinite(sbase.get("roc_auc", np.nan)) else np.nan
                ),
                "delta_regret_aug_minus_base": saug.get("decision_regret_mean", np.nan) - sbase.get("decision_regret_mean", np.nan),
            })
    increment_df = pd.DataFrame(increments)
    increment_df.to_csv(outdir / "p2_2_2_incremental_feature_value.csv", index=False, encoding="utf-8-sig")

    # Detailed false-refresh and missed-benefit examples for primary target.
    primary_errors = pred_df[
        (pred_df.target_variant == args.target) &
        (pred_df.horizon_H.isin(horizons)) &
        (pred_df.error_type != "correct")
    ].copy()
    primary_errors.to_csv(outdir / "p2_2_2_decision_errors.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# SVR-Depth P2.2.2 Report", "",
        f"- Input directory: `{inp}`",
        f"- Primary target: `{args.target}`",
        f"- Min future-pair coverage: {args.min_coverage:.2f}",
        f"- Ridge alpha: {args.ridge_alpha:g}",
        "- Validation: leave one complete sequence out; preprocessors fit only on training sequences.",
        "- Anchor GT is audit-only; valid future labels can be used without anchor GT.",
        "- Horizons crossing ignored restart requests are excluded.",
        "- Decision threshold is fixed at zero; no threshold tuning on test folds.",
        "",
        "## Sample audit", "",
        f"- Candidate rows: {len(audit)}",
        f"- Eligible rows: {len(valid)}",
        f"- Excluded rows: {len(audit)-len(valid)}",
        "",
        "| Sequence | H | Candidates | Eligible |", "|---|---:|---:|---:|",
    ]
    for seq in SEQUENCES:
        for h in horizons:
            sub = audit[(audit.sequence == seq) & (audit.horizon_H == h)]
            lines.append(f"| {seq} | {h} | {len(sub)} | {int(sub.eligible.sum())} |")
    lines += ["", "## Per-sequence primary-target metrics", "",
              "| H | Held-out sequence | Model | N | MAE | Spearman | Balanced acc. | Negative recall | AUC | Regret |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return "NA" if pd.isna(v) else f"{v:.4g}"
    if not metric_df.empty:
        perseq = metric_df[(metric_df.split == "held_out_sequence") &
                           (metric_df.target_variant == args.target)]
        for _, r in perseq.iterrows():
            lines.append(
                f"| {int(r.horizon_H)} | {r.sequence} | {r.model} | {int(r['n'])} | "
                f"{fmt(r.get('mae', np.nan))} | {fmt(r.get('spearman', np.nan))} | "
                f"{fmt(r.get('balanced_accuracy', np.nan))} | {fmt(r.get('negative_recall', np.nan))} | "
                f"{fmt(r.get('roc_auc', np.nan))} | {fmt(r.get('decision_regret_mean', np.nan))} |"
            )
    lines += ["", "## Incremental feature value", "",
              "Delta = augmented model minus base model. Negative delta MAE/regret is better; positive delta Spearman/AUC is better.",
              "", "| Target | H | Sequence | Comparison | N | Delta MAE | Delta Spearman | Delta AUC | Delta regret |",
              "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for _, r in increment_df.iterrows():
        lines.append(
            f"| {r.target_variant} | {int(r.horizon_H)} | {r.sequence} | {r.comparison} | "
            f"{int(r.n_paired)} | {fmt(r.delta_mae_aug_minus_base)} | "
            f"{fmt(r.delta_spearman_aug_minus_base)} | {fmt(r.delta_auc_aug_minus_base)} | "
            f"{fmt(r.delta_regret_aug_minus_base)} |"
        )
    lines += [
        "", "## Interpretation cautions", "",
        "1. Per-sequence results are primary; only three sequences are available.",
        "2. H windows overlap and anchor samples are temporally correlated.",
        "3. `false_refresh` means target <= 0 but predicted > 0; `missed_benefit` means target > 0 but predicted <= 0.",
        "4. These labels measure full-frame state-writeback under forced propagation, not real online B0/B1/B2 utility.",
        "5. Compare the original and discount-weight-normalized target; the latter is a sensitivity analysis.",
        "",
        "## Output files", "",
        "- `p2_2_2_sample_audit.csv`",
        "- `p2_2_2_valid_dataset.csv`",
        "- `p2_2_2_predictions.csv`",
        "- `p2_2_2_metrics.csv`",
        "- `p2_2_2_incremental_feature_value.csv`",
        "- `p2_2_2_decision_errors.csv`",
        "- `p2_2_2_report.md`",
    ]
    (outdir / "p2_2_2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("=== P2.2.2 complete ===")
    print(f"Candidate rows: {len(audit)}; eligible: {len(valid)}; excluded: {len(audit)-len(valid)}")
    print("Outputs:", outdir)
    print("Metrics rows:", len(metric_df), "incremental comparisons:", len(increment_df),
          "decision-error rows:", len(primary_errors))
    print("\nPrimary target pooled metrics:")
    pooled = metric_df[(metric_df.split == "pooled_LOSO") & (metric_df.target_variant == args.target)]
    cols = [c for c in ["horizon_H", "model", "n", "mae", "spearman",
                        "balanced_accuracy", "negative_recall", "roc_auc", "decision_regret_mean"]
            if c in pooled.columns]
    print(pooled[cols].to_string(index=False))


if __name__ == "__main__":
    main()
