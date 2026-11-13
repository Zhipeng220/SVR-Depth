#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.3 v5.17: recover full action-before features and audit temporal proxies.

CPU-only. Joins the P2.2.2 H=16 dataset to valid v5.15 restart-aware labels on
(sequence, anchor_frame, horizon_H), restricted to the auto trajectory. Uses the
v5.15 restart-aware target for LOSO fitting. It does not run VeloDepth inference,
train the depth model, or modify any model weights.
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "G_state_absrel_restart_aware_discounted_per_valid_frame"
OLD_STATE_SCORE = "score_state_only"
OLD_COMBINED_SCORE = "score_motion_plus_learned"
SEQUENCES = ["balloon2", "crowd2", "crowd3"]
KEYS = ["sequence", "anchor_frame", "horizon_H"]
POLICY = "auto"
DEFAULT_ALPHA = 10.0

# Only numeric, action-before feature families are eligible for the expanded set.
FEATURE_PREFIXES = (
    "scheduler_", "image_", "state_features_teacher_", "state_radius_",
    "state_confidence_", "radius_teacher_", "confidence_teacher_", "rgb_",
    "flow_", "luma_", "motion_", "feature_",
)
LEAKAGE_TOKENS = (
    "target", "outcome", "absrel", "rmse", "delta", "future", "label",
    "valid", "status", "action", "score", "gain", "restart", "metric",
    "ground_truth", "gt_", "teacher_depth", "horizon", "sequence",
)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def spearman(a, b):
    x, y = num(pd.Series(a)), num(pd.Series(b))
    m = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    x, y = x[m], y[m]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def pearson(a, b):
    x, y = num(pd.Series(a)), num(pd.Series(b))
    m = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    x, y = x[m], y[m]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="pearson"))


def canonical_sequence(s):
    return s.astype(str).str.strip().str.lower()


def find_anchor_col(df: pd.DataFrame) -> str:
    for c in ["anchor_frame", "anchor", "anchor_idx", "anchor_index", "frame_idx", "frame_index"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot identify anchor column. Expected one of: anchor_frame, anchor, anchor_idx, anchor_index, frame_idx, frame_index")


def load_v515(root: Path):
    paths = sorted(root.rglob("p2_3_v5_15_same_state_anchor_labels.csv"))
    if len(paths) != 3:
        raise RuntimeError(f"Expected exactly 3 v5.15 label CSVs under {root}; found {len(paths)}")
    parts = []
    required = {"sequence", "trajectory_policy", "anchor_frame", "horizon_H", "status", TARGET,
                "feature_scheduler_distance_kf", "feature_scheduler_flow_magnitude",
                "feature_scheduler_total_luma_change", "feature_state_radius_teacher_mean",
                "feature_state_confidence_teacher_mean", OLD_STATE_SCORE, OLD_COMBINED_SCORE}
    for p in paths:
        d = pd.read_csv(p)
        miss = sorted(required - set(d.columns))
        if miss:
            raise ValueError(f"{p}: missing required columns {miss}")
        d["sequence"] = canonical_sequence(d["sequence"])
        d["trajectory_policy"] = d["trajectory_policy"].astype(str).str.strip()
        for c in ["anchor_frame", "horizon_H", TARGET, OLD_STATE_SCORE, OLD_COMBINED_SCORE]:
            d[c] = num(d[c])
        parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    seqs = sorted(raw.sequence.unique().tolist())
    if set(seqs) != set(SEQUENCES):
        raise RuntimeError(f"Expected sequences {SEQUENCES}; found {seqs}")
    d = raw[(raw.status.astype(str).eq("ok")) & (raw.trajectory_policy.eq(POLICY)) & (raw.horizon_H.eq(16))].copy()
    if d.duplicated(KEYS).any():
        raise RuntimeError("Duplicate v5.15 auto/H=16 keys: " + str(d.loc[d.duplicated(KEYS, keep=False), KEYS].head(10).to_dict("records")))
    if not np.isfinite(num(d[TARGET]).to_numpy(float)).all():
        raise RuntimeError("Non-finite restart-aware target in valid v5.15 auto/H=16 rows")
    for score_col in [OLD_STATE_SCORE, OLD_COMBINED_SCORE]:
        if not np.isfinite(num(d[score_col]).to_numpy(float)).all():
            raise RuntimeError(f"Non-finite recorded legacy predictor score in valid rows: {score_col}")
    return raw, d, paths


def load_p222(path: Path):
    d = pd.read_csv(path)
    required = {"sequence", "horizon_H"}
    miss = sorted(required - set(d.columns))
    if miss:
        raise ValueError(f"P2.2.2 dataset missing columns: {miss}")
    anchor_col = find_anchor_col(d)
    if anchor_col != "anchor_frame":
        d = d.rename(columns={anchor_col: "anchor_frame"})
    d["sequence"] = canonical_sequence(d["sequence"])
    d["anchor_frame"] = num(d["anchor_frame"])
    d["horizon_H"] = num(d["horizon_H"])
    d = d[d.horizon_H.eq(16)].copy()
    if d.duplicated(KEYS).any():
        raise RuntimeError("Duplicate P2.2.2 H=16 keys: " + str(d.loc[d.duplicated(KEYS, keep=False), KEYS].head(10).to_dict("records")))
    if not set(d.sequence.unique()).issubset(set(SEQUENCES)):
        raise RuntimeError(f"Unexpected sequence names in P2.2.2 dataset: {sorted(d.sequence.unique())}")
    return d


def eligible_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if c in KEYS or c == "horizon_H":
            continue
        if not cl.startswith(FEATURE_PREFIXES):
            continue
        if any(t in cl for t in LEAKAGE_TOKENS):
            continue
        v = num(df[c])
        if v.notna().sum() >= max(3, int(0.2 * len(df))):
            cols.append(c)
    return sorted(cols)


def choose_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_motion_cols(df):
    candidates = [
        ["scheduler_flow_magnitude"], ["scheduler_total_luma_change"],
        ["image_absdiff_mean_prev"], ["image_absdiff_p90_prev"],
    ]
    return [c for group in candidates if (c := choose_col(df, group))]


def get_state_mean_cols(df):
    out = []
    for cands in [
        ["radius_teacher_mean", "state_radius_teacher_mean"],
        ["confidence_teacher_mean", "state_confidence_teacher_mean"],
    ]:
        c = choose_col(df, cands)
        if c: out.append(c)
    for i in range(4):
        c = choose_col(df, [f"state_features_teacher_{i}_mean", f"feature_state_features_teacher_{i}_mean"])
        if c: out.append(c)
    return out


def get_state_std_cols(df):
    out = []
    for cands in [
        ["radius_teacher_std", "state_radius_teacher_std"],
        ["confidence_teacher_std", "state_confidence_teacher_std"],
    ]:
        c = choose_col(df, cands)
        if c: out.append(c)
    for i in range(4):
        c = choose_col(df, [f"state_features_teacher_{i}_std", f"feature_state_features_teacher_{i}_std"])
        if c: out.append(c)
    return out


def dataset_to_label_feature(col: str, label_cols: set[str]):
    candidates = ["feature_" + col, col]
    aliases = {
        "radius_teacher_mean": ["feature_state_radius_teacher_mean"],
        "state_radius_teacher_mean": ["feature_state_radius_teacher_mean"],
        "confidence_teacher_mean": ["feature_state_confidence_teacher_mean"],
        "state_confidence_teacher_mean": ["feature_state_confidence_teacher_mean"],
    }
    candidates += aliases.get(col, [])
    m = re.fullmatch(r"state_features_teacher_(\d+)_(mean|std)", col)
    if m:
        candidates += [f"feature_state_features_teacher_{m.group(1)}_{m.group(2)}"]
    return next((x for x in candidates if x in label_cols), None)


def feature_alignment(merged: pd.DataFrame, dataset_features: list[str], label_cols: set[str], atol: float, rtol: float):
    long_rows = []
    for dc in dataset_features:
        lc = dataset_to_label_feature(dc, label_cols)
        if not lc:
            continue
        a, b = num(merged[dc]).to_numpy(float), num(merged[lc]).to_numpy(float)
        valid = np.isfinite(a) & np.isfinite(b)
        if not valid.any():
            continue
        diff = np.abs(a[valid] - b[valid])
        close = np.isclose(a[valid], b[valid], atol=atol, rtol=rtol)
        for ix in np.where(valid)[0]:
            long_rows.append({
                "sequence": merged.iloc[ix]["sequence"], "anchor_frame": merged.iloc[ix]["anchor_frame"],
                "dataset_feature": dc, "v515_feature": lc,
                "dataset_value": a[ix], "v515_value": b[ix],
                "abs_diff": abs(a[ix] - b[ix]), "within_tolerance": bool(np.isclose(a[ix], b[ix], atol=atol, rtol=rtol)),
            })
    long = pd.DataFrame(long_rows)
    if long.empty:
        summary = pd.DataFrame(columns=["dataset_feature", "v515_feature", "n_compared", "within_tolerance_n", "within_rate", "mean_abs_diff", "max_abs_diff"])
    else:
        summary = long.groupby(["dataset_feature", "v515_feature"], as_index=False).agg(
            n_compared=("abs_diff", "size"), within_tolerance_n=("within_tolerance", "sum"),
            within_rate=("within_tolerance", "mean"), mean_abs_diff=("abs_diff", "mean"), max_abs_diff=("abs_diff", "max"))
    return long, summary


def time_proxy_audit(merged: pd.DataFrame, age_col: str | None, v515_age_col: str | None):
    rows = []
    for seq, g in merged.groupby("sequence", sort=True):
        frame = num(g["anchor_frame"]).to_numpy(float)
        for source, col in [("p2_2_2", age_col), ("v5.15_online_state", v515_age_col)]:
            if not col or col not in g.columns:
                continue
            age = num(g[col]).to_numpy(float)
            m = np.isfinite(frame) & np.isfinite(age)
            x, y = frame[m], age[m]
            if len(x) < 3:
                continue
            if len(np.unique(x)) > 1:
                model = LinearRegression().fit(x.reshape(-1, 1), y)
                residual = y - model.predict(x.reshape(-1, 1))
            else:
                residual = y - np.mean(y)
            diff = np.diff(y)
            rows.append({
                "sequence": seq, "age_source": source, "age_feature": col, "n": int(len(y)),
                "age_unique": int(len(np.unique(y))), "frame_unique": int(len(np.unique(x))),
                "exact_equal_rate_age_eq_frame": float(np.mean(np.isclose(y, x, atol=1e-9, rtol=0))),
                "spearman_age_vs_frame": spearman(y, x), "pearson_age_vs_frame": pearson(y, x),
                "mean_abs_age_minus_frame": float(np.mean(np.abs(y-x))),
                "age_std": float(np.std(y)), "age_frame_residual_std": float(np.std(residual)),
                "residual_to_age_std_ratio": float(np.std(residual) / np.std(y)) if np.std(y) > 1e-12 else 0.0,
                "negative_age_steps": int(np.sum(diff < -1e-9)), "positive_age_steps": int(np.sum(diff > 1e-9)),
                "zero_age_steps": int(np.sum(np.isclose(diff, 0.0, atol=1e-9))),
            })
    return pd.DataFrame(rows)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], alpha: float):
    y = num(train[TARGET]).to_numpy(float)
    if not features:
        return np.full(len(test), float(np.mean(y))), {"active_features": [], "fallback": "train_mean_no_features"}
    xtr = train[features].apply(pd.to_numeric, errors="coerce").astype(float)
    xte = test[features].apply(pd.to_numeric, errors="coerce").astype(float)
    med = xtr.median(axis=0).fillna(0.0)
    xtr, xte = xtr.fillna(med), xte.fillna(med)
    varying = [c for c in features if xtr[c].nunique(dropna=False) > 1]
    if not varying:
        return np.full(len(test), float(np.mean(y))), {"active_features": [], "fallback": "all_features_constant"}
    scaler = StandardScaler()
    a = scaler.fit_transform(xtr[varying])
    b = scaler.transform(xte[varying])
    model = Ridge(alpha=alpha)
    model.fit(a, y)
    return model.predict(b).astype(float), {"active_features": varying, "fallback": ""}


def evaluate_scores(g: pd.DataFrame, score: np.ndarray, k_fixed: int):
    y_col = "true_gain" if "true_gain" in g.columns else TARGET
    y = num(g[y_col]).to_numpy(float)
    s = np.asarray(score, dtype=float)
    n = len(y)
    out = {"n": n, "target_mean": float(np.mean(y)), "target_positive_rate": float(np.mean(y > 0)),
           "spearman": spearman(s, y), "pearson": pearson(s, y),
           "mae": float(mean_absolute_error(y, s)), "rmse": float(math.sqrt(mean_squared_error(y, s))),
           "r2": float(r2_score(y, s)) if n >= 2 else float("nan"),
           "score_unique": int(pd.Series(s).nunique())}
    if len(np.unique(s)) < 2:
        out.update({"ranking_defined": False, "top25_k": np.nan, "top25_lift": np.nan, "top25_regret": np.nan,
                    "top25_positive_rate": np.nan, "topk_k": np.nan, "topk_lift": np.nan,
                    "topk_regret": np.nan, "topk_positive_rate": np.nan})
        return out
    out["ranking_defined"] = True
    order = np.argsort(-s, kind="stable")
    k25 = max(1, int(math.ceil(n * 0.25)))
    for prefix, k in [("top25", k25), ("topk", min(k_fixed, n))]:
        chosen = order[:k]
        oracle = np.argsort(-y, kind="stable")[:k]
        out[f"{prefix}_k"] = int(k)
        out[f"{prefix}_lift"] = float(np.mean(y[chosen]) - np.mean(y))
        out[f"{prefix}_regret"] = float(np.mean(y[oracle]) - np.mean(y[chosen]))
        out[f"{prefix}_positive_rate"] = float(np.mean(y[chosen] > 0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v515-root", required=True, help="Root containing the three v5.15 per-sequence output folders")
    ap.add_argument("--dataset", required=True, help="P2.2.2 p2_2_2_valid_dataset.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--budget-k", type=int, default=8)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()
    root, dataset_path, out = Path(args.v515_root), Path(args.dataset), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"P2.2.2 dataset not found: {dataset_path}")
    raw, v515, v515_paths = load_v515(root)
    p222 = load_p222(dataset_path)
    p222_features = eligible_feature_cols(p222)

    # Key audit: report overlap and non-overlap in both directions before fitting.
    key_status = v515[KEYS].merge(p222[KEYS], on=KEYS, how="outer", indicator=True)
    key_counts = key_status["_merge"].value_counts().to_dict()
    key_status.to_csv(out / "p2_3_v5_17_key_alignment_audit.csv", index=False)
    merged = v515.merge(p222, on=KEYS, how="inner", suffixes=("_v515", "_p222"), validate="one_to_one")
    if len(merged) < 20:
        raise RuntimeError(f"Only {len(merged)} matched auto/H=16 rows; expected a substantial overlap (historically about 49). Check key semantics before interpreting results.")
    if set(merged.sequence.unique()) != set(SEQUENCES):
        raise RuntimeError(f"Matched dataset does not cover all three sequences: {sorted(merged.sequence.unique())}")

    label_cols = set(v515.columns)
    align_long, align_summary = feature_alignment(merged, p222_features, label_cols, args.atol, args.rtol)
    align_long.to_csv(out / "p2_3_v5_17_feature_alignment_by_sample.csv", index=False)
    align_summary.to_csv(out / "p2_3_v5_17_feature_alignment_summary.csv", index=False)

    age_p222 = choose_col(merged, ["scheduler_distance_kf"])
    age_v515 = choose_col(merged, ["feature_scheduler_distance_kf"])
    motion_cols = get_motion_cols(merged)
    state_mean_cols = get_state_mean_cols(merged)
    state_std_cols = get_state_std_cols(merged)
    state_moment_cols = state_mean_cols + state_std_cols
    full_cols = [c for c in p222_features if c in merged.columns]
    full_no_age_cols = [c for c in full_cols if c != age_p222]
    v516_cols = [c for c in [
        "feature_scheduler_distance_kf", "feature_scheduler_flow_magnitude", "feature_scheduler_total_luma_change",
        "feature_state_radius_teacher_mean", "feature_state_confidence_teacher_mean",
        *[f"feature_state_features_teacher_{i}_mean" for i in range(4)],
    ] if c in merged.columns]
    p222_reduced_cols = [c for c in [age_p222, *motion_cols, *state_mean_cols] if c and c in merged.columns]
    specs = {
        "train_mean_baseline": [],
        "absolute_frame_ridge": ["anchor_frame"],
        "state_age_p222_ridge": [age_p222] if age_p222 else [],
        "state_age_v515_ridge": [age_v515] if age_v515 else [],
        "motion_p222_ridge": motion_cols,
        "state_means_p222_ridge": state_mean_cols,
        "state_moments_p222_ridge": state_moment_cols,
        "reduced_v516_features_from_v515_ridge": v516_cols,
        "reduced_features_from_p222_ridge": p222_reduced_cols,
        "full_action_before_p222_ridge": full_cols,
        "full_action_before_without_age_p222_ridge": full_no_age_cols,
        "full_action_before_plus_frame_p222_ridge": list(dict.fromkeys(full_cols + ["anchor_frame"])),
    }
    # Avoid comparing duplicate feature sets under misleading labels.
    specs = {k: list(dict.fromkeys([c for c in cols if c in merged.columns])) for k, cols in specs.items()}
    legacy = {"legacy_state_score_v515": OLD_STATE_SCORE, "legacy_motion_plus_learned_score_v515": OLD_COMBINED_SCORE}

    time_audit = time_proxy_audit(merged, age_p222, age_v515)
    time_audit.to_csv(out / "p2_3_v5_17_time_proxy_audit.csv", index=False)

    pred_rows, fold_rows = [], []
    for heldout in SEQUENCES:
        train = merged[merged.sequence.ne(heldout)].copy()
        test = merged[merged.sequence.eq(heldout)].copy()
        train_sequences = sorted(train.sequence.unique().tolist())
        if heldout in train_sequences or len(train_sequences) != 2:
            raise RuntimeError(f"LOSO split check failed for {heldout}: {train_sequences}")
        for name, features in specs.items():
            pred, meta = fit_predict(train, test, features, args.alpha)
            fold_rows.append({"heldout_test_sequence": heldout, "model": name,
                              "train_sequences": ";".join(train_sequences), "train_rows": len(train),
                              "test_rows": len(test), "alpha": args.alpha,
                              "candidate_features": ";".join(features),
                              "active_features": ";".join(meta.get("active_features", [])),
                              "fallback": meta.get("fallback", ""), "split_check": "PASS"})
            for (_, row), yhat in zip(test.iterrows(), pred):
                pred_rows.append({"sequence": row.sequence, "anchor_frame": row.anchor_frame, "horizon_H": row.horizon_H,
                                  "heldout_test_sequence": heldout, "model": name, "prediction": float(yhat),
                                  "true_gain": float(row[TARGET]), "train_sequences": ";".join(train_sequences),
                                  "target": TARGET})
        for name, col in legacy.items():
            for _, row in test.iterrows():
                pred_rows.append({"sequence": row.sequence, "anchor_frame": row.anchor_frame, "horizon_H": row.horizon_H,
                                  "heldout_test_sequence": heldout, "model": name, "prediction": float(row[col]),
                                  "true_gain": float(row[TARGET]), "train_sequences": "recorded_in_v5.15",
                                  "target": TARGET})

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out / "p2_3_v5_17_loso_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "p2_3_v5_17_loso_fold_audit.csv", index=False)
    group_rows = []
    for seq, g in pred_df.groupby("sequence", sort=True):
        for model, gm in g.groupby("model", sort=False):
            expected = int((merged.sequence == seq).sum())
            if len(gm) != expected:
                raise RuntimeError(f"Prediction count mismatch for {seq}/{model}: {len(gm)} != {expected}")
            vals = evaluate_scores(gm, gm.prediction.to_numpy(float), args.budget_k)
            group_rows.append({"sequence": seq, "model": model, **vals})
    metrics = pd.DataFrame(group_rows)
    metrics.to_csv(out / "p2_3_v5_17_sequence_metrics.csv", index=False)

    model_order = list(specs) + list(legacy)
    macro_rows = []
    for model in model_order:
        g = metrics[metrics.model.eq(model)]
        row = {"model": model, "sequences": int(len(g)),
               "positive_top25_lift_sequences": int((g.top25_lift > 0).sum()),
               "positive_topk_lift_sequences": int((g.topk_lift > 0).sum()),
               "ranking_defined_sequences": int(g.ranking_defined.fillna(False).sum())}
        for c in ["spearman", "pearson", "mae", "rmse", "r2", "top25_lift", "top25_regret", "topk_lift", "topk_regret", "top25_positive_rate", "topk_positive_rate"]:
            row[f"macro_{c}"] = float(g[c].mean(skipna=True)) if c in g and g[c].notna().any() else float("nan")
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)
    macro.to_csv(out / "p2_3_v5_17_macro_summary.csv", index=False)

    # Paired deltas versus the reduced v5.16 feature set, on the same held-out sequences.
    base_name = "reduced_v516_features_from_v515_ridge"
    base = metrics[metrics.model.eq(base_name)].set_index("sequence") if base_name in set(metrics.model) else pd.DataFrame()
    delta_rows = []
    if not base.empty:
        for model in model_order:
            if model == base_name: continue
            curr = metrics[metrics.model.eq(model)].set_index("sequence")
            for seq in sorted(set(base.index) & set(curr.index)):
                a, b = curr.loc[seq], base.loc[seq]
                delta_rows.append({"sequence": seq, "model": model,
                    "delta_spearman_vs_reduced_v516": a.spearman - b.spearman if pd.notna(a.spearman) and pd.notna(b.spearman) else np.nan,
                    "delta_top25_lift_vs_reduced_v516": a.top25_lift - b.top25_lift if pd.notna(a.top25_lift) and pd.notna(b.top25_lift) else np.nan,
                    "delta_topk_lift_vs_reduced_v516": a.topk_lift - b.topk_lift if pd.notna(a.topk_lift) and pd.notna(b.topk_lift) else np.nan,
                    "delta_mae_vs_reduced_v516": a.mae - b.mae if pd.notna(a.mae) and pd.notna(b.mae) else np.nan,
                    "delta_top25_regret_vs_reduced_v516": a.top25_regret - b.top25_regret if pd.notna(a.top25_regret) and pd.notna(b.top25_regret) else np.nan})
    pd.DataFrame(delta_rows).to_csv(out / "p2_3_v5_17_paired_deltas.csv", index=False)

    # Data and feature audit summaries.
    alignment_overall = {
        "features_compared": int(len(align_summary)),
        "all_values_compared": int(align_summary.n_compared.sum()) if not align_summary.empty else 0,
        "all_values_within_tolerance": int(align_summary.within_tolerance_n.sum()) if not align_summary.empty else 0,
        "weighted_within_tolerance_rate": float(align_summary.within_tolerance_n.sum() / align_summary.n_compared.sum()) if not align_summary.empty and align_summary.n_compared.sum() else None,
        "features_with_any_mismatch": int((align_summary.within_rate < 1.0).sum()) if not align_summary.empty else 0,
    }
    meta = {
        "experiment": "P2.3 v5.17",
        "title": "Full action-before feature recovery and temporal-proxy ablation audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"v515_root": str(root), "v515_label_files": [str(x) for x in v515_paths], "p2_2_2_dataset": str(dataset_path)},
        "target": TARGET, "horizon_H": 16, "trajectory_policy": POLICY,
        "matched_rows": int(len(merged)), "matched_rows_by_sequence": merged.groupby("sequence").size().to_dict(),
        "key_overlap_counts": {str(k): int(v) for k, v in key_counts.items()},
        "p2_2_2_h16_rows": int(len(p222)), "v515_valid_auto_h16_rows": int(len(v515)),
        "p2_2_2_candidate_action_before_features": p222_features,
        "feature_alignment_summary": alignment_overall,
        "feature_groups": specs, "alpha": args.alpha, "budget_k": args.budget_k,
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
        "limitations": [
            "Only the auto trajectory is used for feature recovery because P2.2.2 is the historical auto/H=16 training dataset; do not copy its state features to other policy trajectories.",
            "The sample has only three sequences; LOSO is a strict exploratory test, not a broad generalization proof.",
            "The restart-aware target executes future automatic restarts but does not replay future proactive policy refreshes; it is not a complete online-policy counterfactual.",
            "Recorded legacy predictor scores were trained on the earlier target and are evaluated only as migration baselines.",
        ],
    }
    (out / "p2_3_v5_17_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable report generated from the actual run outputs.
    def fmt(v, digits=4):
        return "NA" if pd.isna(v) else f"{float(v):.{digits}f}"
    lines = [
        "# P2.3 v5.17 实验报告：完整动作前特征恢复与时间代理消融审计", "",
        "## 1. 实验范围", "",
        f"- 目标：`{TARGET}`（H=16，重启感知收益）。",
        f"- 轨迹：仅 `auto`；P2.2.2 数据集与 v5.15 标签按 `{', '.join(KEYS)}` 对齐。",
        f"- 匹配样本：{len(merged)}；P2.2.2 H=16 行数：{len(p222)}；v5.15 有效 auto/H=16 行数：{len(v515)}。",
        "- 本轮只进行 CPU 离线分析，不运行 VeloDepth 推理、不训练主模型、不修改权重。", "",
        "## 2. 对齐核验", "",
        f"- 键匹配：both={key_counts.get('both', 0)}，仅 v5.15={key_counts.get('left_only', 0)}，仅 P2.2.2={key_counts.get('right_only', 0)}。",
        f"- 有 {alignment_overall['features_compared']} 个共享特征字段可逐样本比较，共 {alignment_overall['all_values_compared']} 个数值对。",
        f"- 容差内比例：{fmt(alignment_overall['weighted_within_tolerance_rate'], 4)}；存在至少一个不一致值的字段数：{alignment_overall['features_with_any_mismatch']}。", "",
        "特征对齐的容差为 `atol=" + str(args.atol) + "`, `rtol=" + str(args.rtol) + "`。若存在不一致，不能直接将离线特征当作在线动作前特征的等价替代；应查看 `p2_3_v5_17_feature_alignment_summary.csv` 和逐样本差异。", "",
        "## 3. 时间代理审计", "",
        "| 序列 | 来源 | 年龄特征 | age=frame 比例 | Spearman(age, frame) | 残差/年龄标准差 | 年龄下降步数 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    if time_audit.empty:
        lines.append("| — | — | — | — | — | — | — |")
    else:
        for _, r in time_audit.iterrows():
            lines.append(f"| {r.sequence} | {r.age_source} | `{r.age_feature}` | {fmt(r.exact_equal_rate_age_eq_frame)} | {fmt(r.spearman_age_vs_frame)} | {fmt(r.residual_to_age_std_ratio)} | {int(r.negative_age_steps)} |")
    lines += ["", "残差/年龄标准差接近 0 表示年龄几乎可以由帧位置线性解释；这只是时间混淆诊断，不是因果证明。", "", "## 4. LOSO 预测器比较", "",
              "每折完整留出一条序列，其他两条序列用于拟合。指标先在每条留出序列内计算，再对三条序列等权宏平均。Top-k 使用每条留出序列的最多 8 个锚点；不是完整在线候选集的同预算策略结果。", "",
              "| 模型 | Spearman ↑ | MAE ↓ | Top25 lift ↑ | Top8 lift ↑ | Top8 regret ↓ | 正向 Top25 序列数 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in macro.iterrows():
        lines.append(f"| {r.model} | {fmt(r.macro_spearman)} | {fmt(r.macro_mae)} | {fmt(r.macro_top25_lift)} | {fmt(r.macro_topk_lift)} | {fmt(r.macro_topk_regret)} | {int(r.positive_top25_lift_sequences)}/3 |")
    lines += ["", "## 5. 解释边界", "",
              "1. 本轮只对 `auto` 轨迹恢复完整特征；不能将 P2.2.2 状态特征复制给另外三种策略，因为 v5.14 已显示同锚点状态特征可能随策略状态改变。",
              "2. 只有在匹配覆盖充分、键语义正确、共享特征数值一致性可接受时，完整特征模型与精简特征模型的差异才可解释为特征扩展的影响。",
              "3. 若年龄模型仍领先，但年龄与帧位置高度共线，则不能把其效果直接归因于真实状态年龄。",
              "4. 三条序列的小样本 LOSO 仅用于诊断，不能证明广泛泛化。",
              "5. 重启感知标签未重放未来主动策略刷新，因此仍不能替代完整在线策略的反事实评估。", "",
              "## 6. 产物", "",
              "- `p2_3_v5_17_key_alignment_audit.csv`：匹配键审计。",
              "- `p2_3_v5_17_feature_alignment_by_sample.csv`：共享特征逐样本差异。",
              "- `p2_3_v5_17_feature_alignment_summary.csv`：共享特征一致性汇总。",
              "- `p2_3_v5_17_time_proxy_audit.csv`：年龄与帧位置审计。",
              "- `p2_3_v5_17_loso_predictions.csv`：逐样本预测。",
              "- `p2_3_v5_17_loso_fold_audit.csv`：LOSO 隔离与特征组审计。",
              "- `p2_3_v5_17_sequence_metrics.csv`：逐序列指标。",
              "- `p2_3_v5_17_macro_summary.csv`：三序列宏平均。",
              "- `p2_3_v5_17_paired_deltas.csv`：相对 v5.16 精简特征组的配对差值。",
              "- `p2_3_v5_17_metadata.json`：输入、参数、覆盖率和限制。", ""]
    (out / "P2_3_v5_17_实验报告.md").write_text("\n".join(lines), encoding="utf-8")

    print("P2.3 v5.17 complete")
    print(f"matched_rows={len(merged)}; key_overlap={key_counts}; candidate_full_features={len(full_cols)}")
    print(f"feature_alignment={alignment_overall}")
    print(f"outputs={out.resolve()}")
    print(macro[["model", "macro_spearman", "macro_mae", "macro_top25_lift", "macro_topk_lift", "macro_topk_regret"]].to_string(index=False))

if __name__ == "__main__":
    main()
