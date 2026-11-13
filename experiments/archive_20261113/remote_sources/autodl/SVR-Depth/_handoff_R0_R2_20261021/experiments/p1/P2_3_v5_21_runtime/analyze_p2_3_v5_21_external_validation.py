#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.3 v5.21 locked-model cross-sequence replication + offline fixed-budget audit.

The main candidate is locked from v5.19: state_features_teacher_2_mean,
residualized against scheduler_distance_kf and fit with Ridge(alpha=10).
The model is trained on the three existing sequences only and tested on one or
more held-out sequence label CSVs produced by the v5.21 auto-only
collector. External targets must be v5.15 restart-aware auto/H=16 labels.

This is an offline fixed-K ranking audit, NOT a full online-policy replay or a
latency/compute-budget benchmark. Future H=16 windows overlap, so metrics are
exploratory and do not establish statistical significance.
"""
from __future__ import annotations
import argparse, hashlib, json, math, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "G_state_absrel_restart_aware_discounted_per_valid_frame"
AGE = "scheduler_distance_kf"
FRAME = "anchor_frame"
STATE2 = "state_features_teacher_2_mean"
RADIUS = "state_radius_teacher_mean"
KNOWN_SEQUENCES = {"balloon2", "crowd2", "crowd3"}
ALPHA = 10.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def num(s):
    return pd.to_numeric(s, errors="coerce")


def spearman(a, b):
    x, y = num(pd.Series(a)), num(pd.Series(b))
    mask = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    x, y = x[mask], y[mask]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def evaluate(y, pred, budget_k=8):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = len(y)
    out = {
        "n": int(n), "target_mean": float(np.mean(y)),
        "spearman": spearman(pred, y),
        "pearson": float(pd.Series(pred).corr(pd.Series(y), method="pearson")) if n > 1 and np.std(pred) > 0 and np.std(y) > 0 else float("nan"),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if n > 1 else float("nan"),
    }
    order = np.argsort(-pred, kind="stable")
    oracle_order = np.argsort(-y, kind="stable")
    for tag, k in [("top25", max(1, int(math.ceil(n * 0.25)))), ("top8", min(int(budget_k), n))]:
        chosen = order[:k]
        oracle = oracle_order[:k]
        lift = float(np.mean(y[chosen]) - np.mean(y))
        regret = float(np.mean(y[oracle]) - np.mean(y[chosen]))
        out[f"{tag}_k"] = int(k)
        out[f"{tag}_lift"] = 0.0 if abs(lift) < 1e-12 else lift
        out[f"{tag}_regret"] = 0.0 if abs(regret) < 1e-12 else regret
        out[f"{tag}_selected_mean_gain"] = float(np.mean(y[chosen]))
        out[f"{tag}_oracle_mean_gain"] = float(np.mean(y[oracle]))
        out[f"{tag}_chosen_indices"] = chosen.tolist()
    return out


def fit_age_and_candidate(train: pd.DataFrame, test: pd.DataFrame, feature: str, alpha: float):
    """Fit y~age plus a residual feature model using training data only."""
    ytr = train[TARGET].to_numpy(float)
    age_tr = train[[AGE]].to_numpy(float)
    age_te = test[[AGE]].to_numpy(float)
    base = LinearRegression().fit(age_tr, ytr)
    base_tr = base.predict(age_tr)
    base_te = base.predict(age_te)

    xtr_raw = num(train[feature]).to_numpy(float)
    xte_raw = num(test[feature]).to_numpy(float)
    med = float(np.nanmedian(xtr_raw)) if np.isfinite(xtr_raw).any() else 0.0
    xtr_raw = np.where(np.isfinite(xtr_raw), xtr_raw, med)
    xte_raw = np.where(np.isfinite(xte_raw), xte_raw, med)
    x_age = LinearRegression().fit(age_tr, xtr_raw)
    xres_tr = xtr_raw - x_age.predict(age_tr)
    xres_te = xte_raw - x_age.predict(age_te)
    if float(np.std(xres_tr)) < 1e-10:
        return base_te, {"feature": feature, "fallback": "feature_residual_constant", "ridge_coefficient": 0.0,
                         "age_intercept": float(base.intercept_), "age_slope": float(base.coef_[0]),
                         "feature_median_train": med}
    scaler = StandardScaler().fit(xres_tr.reshape(-1, 1))
    ztr = scaler.transform(xres_tr.reshape(-1, 1))
    zte = scaler.transform(xres_te.reshape(-1, 1))
    yres = ytr - base_tr
    model = Ridge(alpha=alpha).fit(ztr, yres)
    pred = base_te + model.predict(zte)
    return pred, {"feature": feature, "fallback": "", "ridge_coefficient": float(model.coef_[0]),
                  "age_intercept": float(base.intercept_), "age_slope": float(base.coef_[0]),
                  "feature_median_train": med, "ridge_alpha": float(alpha)}


def fit_age_and_candidates(train: pd.DataFrame, test: pd.DataFrame, features: list[str], alpha: float):
    """Secondary multi-feature residual model; not the primary confirmatory model."""
    ytr = train[TARGET].to_numpy(float)
    age_tr = train[[AGE]].to_numpy(float)
    age_te = test[[AGE]].to_numpy(float)
    base = LinearRegression().fit(age_tr, ytr)
    base_tr, base_te = base.predict(age_tr), base.predict(age_te)
    xtr = train[features].apply(pd.to_numeric, errors="coerce").astype(float)
    xte = test[features].apply(pd.to_numeric, errors="coerce").astype(float)
    med = xtr.median().fillna(0.0)
    xtr, xte = xtr.fillna(med), xte.fillna(med)
    xage = LinearRegression().fit(age_tr, xtr.to_numpy(float))
    xr_tr = xtr.to_numpy(float) - xage.predict(age_tr)
    xr_te = xte.to_numpy(float) - xage.predict(age_te)
    keep = [i for i in range(xr_tr.shape[1]) if np.std(xr_tr[:, i]) > 1e-10]
    if not keep:
        return base_te, {"features": ";".join(features), "fallback": "all_residual_features_constant", "coefficients": {}}
    scaler = StandardScaler().fit(xr_tr[:, keep])
    ztr, zte = scaler.transform(xr_tr[:, keep]), scaler.transform(xr_te[:, keep])
    ridge = Ridge(alpha=alpha).fit(ztr, ytr - base_tr)
    pred = base_te + ridge.predict(zte)
    return pred, {"features": ";".join(features), "fallback": "", "coefficients": {features[i]: float(c) for i, c in zip(keep, ridge.coef_)}, "ridge_alpha": float(alpha)}


def load_training(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["sequence", "trajectory_policy", "horizon_H", "status", FRAME, AGE, STATE2, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Training data missing columns: {missing}")
    df["sequence"] = df.sequence.astype(str).str.strip().str.lower()
    df["trajectory_policy"] = df.trajectory_policy.astype(str).str.strip().str.lower()
    df["status"] = df.status.astype(str).str.strip().str.lower()
    df = df[(df.trajectory_policy == "auto") & (num(df.horizon_H) == 16) & (df.status == "ok")].copy()
    for c in [FRAME, AGE, STATE2, TARGET] + ([RADIUS] if RADIUS in df.columns else []):
        df[c] = num(df[c])
    df = df[np.isfinite(df[TARGET]) & np.isfinite(df[AGE]) & np.isfinite(df[STATE2])].copy()
    if df.duplicated(["sequence", FRAME]).any():
        raise ValueError("Training dataset has duplicate sequence+anchor rows")
    seqs = set(df.sequence)
    if not KNOWN_SEQUENCES.issubset(seqs):
        raise ValueError(f"Training dataset must contain the locked source sequences {sorted(KNOWN_SEQUENCES)}; found {sorted(seqs)}")
    if len(df) != 53:
        raise ValueError(f"Expected 53 valid source training anchors from v5.19; got {len(df)}. Do not silently change training set.")
    return df.sort_values(["sequence", FRAME]).reset_index(drop=True)


def load_external_label(path: Path, known_sequences: set[str], min_valid: int, min_coverage: float):
    raw = pd.read_csv(path)
    required = ["sequence", "trajectory_policy", FRAME, "horizon_H", "status",
                "feature_scheduler_distance_kf", "feature_" + STATE2, TARGET,
                "future_frames_requested", "future_metric_pairs"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"External label file {path} missing required columns: {missing}")
    raw["sequence"] = raw.sequence.astype(str).str.strip().str.lower()
    raw["trajectory_policy"] = raw.trajectory_policy.astype(str).str.strip().str.lower()
    raw["status"] = raw.status.astype(str).str.strip().str.lower()
    raw_h16 = raw[(raw.trajectory_policy == "auto") & (num(raw.horizon_H) == 16)].copy()
    ok = raw_h16[raw_h16.status == "ok"].copy()
    if ok.empty:
        raise ValueError(f"No status=ok auto/H=16 labels in {path}")
    seqs = set(ok.sequence)
    if len(seqs) != 1:
        raise ValueError(f"Each external CSV must represent exactly one sequence; found {sorted(seqs)} in {path}")
    seq = next(iter(seqs))
    if seq in known_sequences:
        raise ValueError(f"External sequence '{seq}' is one of the training sequences. Renaming a known sequence is not independent validation.")
    if ok.duplicated(["sequence", FRAME]).any():
        raise ValueError(f"Duplicate sequence+anchor labels in {path}")
    feature_map = {
        AGE: "feature_scheduler_distance_kf",
        STATE2: "feature_state_features_teacher_2_mean",
        RADIUS: "feature_state_radius_teacher_mean",
    }
    for c in [FRAME, TARGET, *[v for k, v in feature_map.items() if v in ok.columns]]:
        ok[c] = num(ok[c])
    before_numeric = len(ok)
    ok = ok[np.isfinite(ok[TARGET]) & np.isfinite(ok[FRAME]) & np.isfinite(ok[feature_map[AGE]]) & np.isfinite(ok[feature_map[STATE2]])].copy()
    if len(ok) < min_valid:
        raise ValueError(f"Sequence {seq}: only {len(ok)} finite valid H=16 anchors after filtering; require >= {min_valid}")
    if len(ok) < 8:
        raise ValueError(f"Sequence {seq}: fewer than 8 valid anchors; cannot evaluate fixed K=8")
    if "future_frames_requested" in ok.columns and not num(ok.future_frames_requested).eq(16).all():
        raise ValueError(f"Sequence {seq}: not all retained labels requested a full H=16 window")
    coverage = float(num(ok.future_metric_pairs).div(num(ok.future_frames_requested)).mean()) if {"future_metric_pairs", "future_frames_requested"}.issubset(ok.columns) else float("nan")
    if np.isfinite(coverage) and coverage < min_coverage:
        raise ValueError(f"Sequence {seq}: mean paired future-metric coverage {coverage:.3f} < required {min_coverage:.3f}")
    canonical = pd.DataFrame({
        "sequence": ok.sequence.to_numpy(), "trajectory_policy": "auto", "anchor_frame": ok[FRAME].to_numpy(float),
        "scheduler_distance_kf": num(ok[feature_map[AGE]]).to_numpy(float),
        "state_features_teacher_2_mean": num(ok[feature_map[STATE2]]).to_numpy(float),
        "G_state_absrel_restart_aware_discounted_per_valid_frame": num(ok[TARGET]).to_numpy(float),
        "horizon_H": 16, "status": "ok", "source_label_csv": str(path),
    })
    if feature_map[RADIUS] in ok.columns:
        canonical["state_radius_teacher_mean"] = num(ok[feature_map[RADIUS]]).to_numpy(float)
    audit = {
        "source_file": str(path), "sequence": seq, "raw_rows": int(len(raw)),
        "auto_h16_rows": int(len(raw_h16)), "status_ok_rows": int(len(raw_h16[raw_h16.status == "ok"])),
        "finite_rows_before_numeric_filter": int(before_numeric), "valid_rows_used": int(len(canonical)),
        "rows_dropped_for_missing_nonfinite_core_fields": int(before_numeric - len(canonical)),
        "mean_future_metric_pair_coverage": coverage,
        "mean_future_metric_pairs": float(num(ok.future_metric_pairs).mean()) if "future_metric_pairs" in ok.columns else float("nan"),
        "unique_anchor_count": int(canonical.anchor_frame.nunique()),
        "anchor_min": float(canonical.anchor_frame.min()), "anchor_max": float(canonical.anchor_frame.max()),
        "age_min": float(canonical.scheduler_distance_kf.min()), "age_max": float(canonical.scheduler_distance_kf.max()),
        "feature_state2_min": float(canonical.state_features_teacher_2_mean.min()),
        "feature_state2_max": float(canonical.state_features_teacher_2_mean.max()),
        "sequence_name_independent_check": "PASS_NOT_IN_LOCKED_TRAINING_SET; HISTORICAL_EXPOSURE_RECORDED",
        "target_protocol": str(ok.future_protocol.iloc[0]) if "future_protocol" in ok.columns else "not_recorded",
    }
    return canonical.sort_values("anchor_frame").reset_index(drop=True), audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dataset", default=None, help="Locked v5.18 aligned 53-anchor source dataset")
    ap.add_argument("--external-label-csv", action="append", default=[], help="Path to a new sequence's v5.15 label CSV; repeat for multiple sequences")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--alpha", type=float, default=ALPHA, help="Locked Ridge alpha; do not tune on external sequence")
    ap.add_argument("--budget-k", type=int, default=8)
    ap.add_argument("--min-valid-anchors", type=int, default=12)
    ap.add_argument("--min-mean-pair-coverage", type=float, default=0.90)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    script_dir = Path(__file__).resolve().parent
    train_path = Path(args.train_dataset) if args.train_dataset else script_dir / "inputs" / "p2_3_v5_18_aligned_53_anchor_dataset.csv"
    out = Path(args.out_dir) if args.out_dir else script_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        # Unit smoke-test for metrics only; not research evidence.
        y = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2])
        m = evaluate(y, y, 8)
        assert m["top8_regret"] == 0.0 and m["mae"] == 0.0
        print("PASS: v5.21 evaluator metric self-test (synthetic fixture only; no research result).")
        return
    if not args.external_label_csv:
        ap.error("At least one --external-label-csv is required unless --self-test is used")
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    train = load_training(train_path)
    known = set(train.sequence)
    ext_paths = [Path(x).resolve() for x in args.external_label_csv]
    if len(set(ext_paths)) != len(ext_paths):
        raise ValueError("The same external label CSV was supplied more than once")
    external_frames, audits = [], []
    external_names = []
    for p in ext_paths:
        if not p.is_file(): raise FileNotFoundError(p)
        frame, audit = load_external_label(p, known, args.min_valid_anchors, args.min_mean_pair_coverage)
        if audit["sequence"] in external_names:
            raise ValueError(f"External sequence '{audit['sequence']}' was supplied more than once")
        external_names.append(audit["sequence"])
        external_frames.append(frame)
        audits.append(audit)
    if not external_frames:
        raise ValueError("At least one external sequence label CSV is required")

    model_audit, pred_rows, metric_rows, delta_rows, selection_rows = [], [], [], [], []
    for ext, audit, source_path in zip(external_frames, audits, ext_paths):
        seq = audit["sequence"]
        # Snapshot external source and fixed training data for reproducibility.
        snapdir = out / "input_snapshots"
        snapdir.mkdir(exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in seq)
        shutil.copy2(source_path, snapdir / f"{safe_name}_v5_15_labels.csv")
        ext.to_csv(out / f"{safe_name}_aligned_external_anchors.csv", index=False)
        ytr = train[TARGET].to_numpy(float)
        yte = ext[TARGET].to_numpy(float)
        model_predictions = {}
        model_predictions["train_mean_baseline"] = np.full(len(ext), float(np.mean(ytr)))
        frame_model = LinearRegression().fit(train[[FRAME]].to_numpy(float), ytr)
        model_predictions["absolute_frame_ols"] = frame_model.predict(ext[[FRAME]].to_numpy(float))
        age_model = LinearRegression().fit(train[[AGE]].to_numpy(float), ytr)
        model_predictions["state_age_ols"] = age_model.predict(ext[[AGE]].to_numpy(float))
        fit_meta = {
            "train_rows": int(len(train)), "train_sequences": ";".join(sorted(known)),
            "external_sequence": seq, "alpha": float(args.alpha), "budget_k": int(args.budget_k),
            "primary_model": "age_resid_state2_mean", "test_sequence_used_for_fit_or_tuning": False,
            "training_age_min": float(train[AGE].min()), "training_age_max": float(train[AGE].max()),
            "external_age_min": float(ext[AGE].min()), "external_age_max": float(ext[AGE].max()),
            "external_age_out_of_training_range_fraction": float(((ext[AGE] < train[AGE].min()) | (ext[AGE] > train[AGE].max())).mean()),
        }
        state2_pred, state2_meta = fit_age_and_candidate(train, ext, STATE2, args.alpha)
        model_predictions["age_resid_state2_mean"] = state2_pred
        model_audit.append({**fit_meta, "model": "age_resid_state2_mean", **{k: v for k, v in state2_meta.items() if not isinstance(v, dict)}})
        if RADIUS in train.columns and RADIUS in ext.columns and ext[RADIUS].notna().any():
            radius_pred, radius_meta = fit_age_and_candidate(train, ext, RADIUS, args.alpha)
            model_predictions["age_resid_radius_mean_negative_control"] = radius_pred
            model_audit.append({**fit_meta, "model": "age_resid_radius_mean_negative_control", **{k: v for k, v in radius_meta.items() if not isinstance(v, dict)}})
            both_pred, both_meta = fit_age_and_candidates(train, ext, [STATE2, RADIUS], args.alpha)
            model_predictions["age_resid_state2_plus_radius_secondary"] = both_pred
            model_audit.append({**fit_meta, "model": "age_resid_state2_plus_radius_secondary", "features": both_meta.get("features"), "fallback": both_meta.get("fallback"), "coefficients_json": json.dumps(both_meta.get("coefficients", {}), ensure_ascii=False), "ridge_alpha": args.alpha})

        for model_name, pred in model_predictions.items():
            if len(pred) != len(ext) or not np.isfinite(pred).all():
                raise RuntimeError(f"Non-finite/wrong-length predictions for {seq}/{model_name}")
            for i, row in ext.iterrows():
                pred_rows.append({"sequence": seq, "anchor_frame": row.anchor_frame, "model": model_name,
                                  "prediction": float(pred[i]), "true_gain": float(row[TARGET]),
                                  "train_sequences": ";".join(sorted(known)), "is_external_test": True})
            m = evaluate(yte, pred, args.budget_k)
            chosen8 = m["top8_chosen_indices"]
            oracle8 = m["top8_chosen_indices"]
            chosen25 = m["top25_chosen_indices"]
            for tag, chosen in [("top8", chosen8), ("top25", chosen25)]:
                for rank, idx in enumerate(chosen, 1):
                    selection_rows.append({"sequence": seq, "model": model_name, "budget": tag,
                                           "rank": rank, "anchor_frame": float(ext.iloc[idx].anchor_frame),
                                           "predicted_gain": float(pred[idx]), "actual_gain": float(yte[idx]),
                                           "is_oracle": False})
            for rank, idx in enumerate(np.argsort(-yte, kind="stable")[:m["top8_k"]], 1):
                selection_rows.append({"sequence": seq, "model": "oracle_top8_reference", "budget": "top8",
                                       "rank": rank, "anchor_frame": float(ext.iloc[idx].anchor_frame),
                                       "predicted_gain": float(yte[idx]), "actual_gain": float(yte[idx]),
                                       "is_oracle": True})
            metric_rows.append({"sequence": seq, "model": model_name, **{k: v for k, v in m.items() if not k.endswith("_chosen_indices")}})
        metrics_seq = {r["model"]: r for r in metric_rows if r["sequence"] == seq}
        base = metrics_seq["state_age_ols"]
        cand = metrics_seq["age_resid_state2_mean"]
        base_pred = model_predictions["state_age_ols"]
        cand_pred = model_predictions["age_resid_state2_mean"]
        k = min(args.budget_k, len(ext))
        base_idx = set(np.argsort(-base_pred, kind="stable")[:k].tolist())
        cand_idx = set(np.argsort(-cand_pred, kind="stable")[:k].tolist())
        base_mean = float(np.mean(yte[list(base_idx)]))
        cand_mean = float(np.mean(yte[list(cand_idx)]))
        delta_rows.append({
            "sequence": seq, "candidate_model": "age_resid_state2_mean", "baseline_model": "state_age_ols",
            "mae_improvement": float(base["mae"] - cand["mae"]),
            "spearman_delta": float(cand["spearman"] - base["spearman"]),
            "top25_regret_improvement": float(base["top25_regret"] - cand["top25_regret"]),
            "top8_regret_improvement": float(base["top8_regret"] - cand["top8_regret"]),
            "top8_lift_delta": float(cand["top8_lift"] - base["top8_lift"]),
            "top8_selection_overlap_count": int(len(base_idx & cand_idx)),
            "top8_selection_overlap_rate": float(len(base_idx & cand_idx) / max(1, k)),
            "age_baseline_selected_mean_gain": base_mean,
            "candidate_selected_mean_gain": cand_mean,
            "selected_mean_gain_delta": cand_mean - base_mean,
            "mae_improved": bool(cand["mae"] < base["mae"]),
            "spearman_improved": bool(cand["spearman"] > base["spearman"]),
            "top8_regret_not_worse": bool(cand["top8_regret"] <= base["top8_regret"] + 1e-12),
            "top8_selected_mean_not_worse": bool(cand_mean >= base_mean - 1e-12),
            "preliminary_gate_pass": bool(cand["mae"] < base["mae"] and cand["spearman"] > base["spearman"] and cand["top8_regret"] <= base["top8_regret"] + 1e-12 and cand_mean >= base_mean - 1e-12),
        })

    pred_df = pd.DataFrame(pred_rows)
    met_df = pd.DataFrame(metric_rows)
    delta_df = pd.DataFrame(delta_rows)
    audit_df = pd.DataFrame(audits)
    pred_df.to_csv(out / "p2_3_v5_21_external_predictions.csv", index=False)
    met_df.to_csv(out / "p2_3_v5_21_external_sequence_metrics.csv", index=False)
    delta_df.to_csv(out / "p2_3_v5_21_fixed_budget_delta_vs_age.csv", index=False)
    audit_df.to_csv(out / "p2_3_v5_21_external_data_audit.csv", index=False)
    pd.DataFrame(model_audit).to_csv(out / "p2_3_v5_21_model_fit_audit.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out / "p2_3_v5_21_selected_anchors.csv", index=False)

    summary_rows = []
    for model, group in met_df.groupby("model", sort=False):
        r = {"model": model, "external_sequence_count": int(group.sequence.nunique())}
        for c in ["spearman", "mae", "rmse", "r2", "top25_lift", "top25_regret", "top8_lift", "top8_regret"]:
            r["macro_" + c] = float(group[c].mean(skipna=True))
        summary_rows.append(r)
    pd.DataFrame(summary_rows).to_csv(out / "p2_3_v5_21_external_macro_summary.csv", index=False)

    # Persist exact inputs and hashes.
    shutil.copy2(train_path, out / "input_snapshots" / "locked_training_dataset.csv")
    metadata = {
        "experiment": "P2.3 v5.21 locked-model cross-sequence replication and offline fixed-budget audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_dataset": str(train_path.resolve()), "training_dataset_sha256": sha256(train_path),
        "training_rows": int(len(train)), "training_sequences": sorted(known),
        "external_sequences": external_names,
        "external_label_files": [{"path": str(p), "sha256": sha256(p)} for p in ext_paths],
        "target": TARGET, "target_protocol": "v5.15 auto/H=16 restart-aware same-state label; future automatic restarts executed independently in B0/B1; future proactive policy actions are not replayed",
        "primary_model": "state_age_ols vs age_resid_state2_mean",
            "historical_sequence_exposure": "synchronous was used in earlier P2.3 v5.10/v5.11 experiments; this is a locked-model replication, not a pristine project-wide unseen-sequence test.",
        "locked_candidate_feature": STATE2, "negative_control_feature": RADIUS,
        "ridge_alpha": float(args.alpha), "fixed_budget_k": int(args.budget_k),
        "minimum_valid_anchors": int(args.min_valid_anchors), "minimum_mean_pair_coverage": float(args.min_mean_pair_coverage),
        "test_sequence_used_for_fit_or_tuning": False,
        "evaluation_scope": "offline candidate ranking only; not full online action replay, not equal measured compute, not end-to-end latency",
        "limitations": [
            "The synchronous sequence was used in earlier P2.3 v5.10/v5.11 experiments; this is a locked-model replication, not a pristine project-wide unseen-sequence test.",
            "Only one held-out sequence is evaluated here; this is not broad generalization evidence.",
            "H=16 future windows overlap across anchors; ordinary per-anchor significance claims are not justified.",
            "Candidate feature was discovered on the original three sequences; external labels are needed for independent confirmation.",
            "This target measures restart-aware same-state writeback benefit, not the complete benefit of a future proactive refresh policy.",
        ],
    }
    (out / "p2_3_v5_21_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable report.
    fmt = lambda v: "NA" if pd.isna(v) else f"{float(v):.5f}"
    lines = [
        "# P2.3 v5.21 实验报告：独立序列验证与固定预算离线决策审计", "",
        "## 1. 实验定位", "", "项目历史记录显示 synchronous 已用于 P2.3 v5.10/v5.11，因此本轮是锁定模型跨序列复核，不是全项目范围内的全新独立测试。", "",
        "训练集固定为 v5.18/v5.19 的 53 个锚点（balloon2、crowd2、crowd3）；主候选固定为 `state_features_teacher_2_mean`，通过状态年龄残差化后用 Ridge(alpha=10) 预测重启感知 H=16 收益。测试序列完全不参与拟合、标准化、填补或参数选择。",
        "本报告中的 Top 8 是离线候选排序审计，不是完整在线策略、实测等算力预算或端到端延迟结果。", "",
        "## 2. 数据审计", "",
        f"- 训练样本：{len(train)}；训练序列：{', '.join(sorted(known))}。",
        f"- 外部序列：{', '.join(external_names)}。",
        f"- 外部有效锚点总数：{sum(int(a['valid_rows_used']) for a in audits)}。",
        f"- 每个序列至少 {args.min_valid_anchors} 个有效 auto/H=16 标签，且平均未来配对指标覆盖率至少 {args.min_mean_pair_coverage:.0%}。",
        "- 外部标签必须是 `status=ok`、`trajectory_policy=auto`、`horizon_H=16`，目标必须与训练集同义。", "",
        "### 外部数据覆盖", "",
        "| 序列 | 有效锚点 | 平均未来指标配对覆盖率 | 状态年龄范围 | 通道 2 范围 |",
        "|---|---:|---:|---:|---:|",
    ]
    for a in audits:
        cov = fmt(a["mean_future_metric_pair_coverage"])
        lines.append(f"| {a['sequence']} | {a['valid_rows_used']} | {cov} | {fmt(a['age_min'])}–{fmt(a['age_max'])} | {fmt(a['feature_state2_min'])}–{fmt(a['feature_state2_max'])} |")
    lines += ["", "## 3. 外部序列指标", "", "所有指标逐序列报告；宏平均仅在提供多个独立外部序列时作为描述性汇总。", "",
              "| 外部序列 | 模型 | Spearman ↑ | MAE ↓ | Top 25% regret ↓ | Top 8 regret ↓ | Top 8 lift ↑ |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in met_df.iterrows():
        lines.append(f"| {r['sequence']} | {r['model']} | {fmt(r['spearman'])} | {fmt(r['mae'])} | {fmt(r['top25_regret'])} | {fmt(r['top8_regret'])} | {fmt(r['top8_lift'])} |")
    lines += ["", "## 4. 主候选相对状态年龄基线", "",
              "| 序列 | MAE 改善 | Spearman 差值 | Top 25% regret 改善 | Top 8 regret 改善 | Top 8 重叠 | Top 8 选择均值收益差 | 初步门槛 |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in delta_df.iterrows():
        lines.append(f"| {r['sequence']} | {fmt(r['mae_improvement'])} | {fmt(r['spearman_delta'])} | {fmt(r['top25_regret_improvement'])} | {fmt(r['top8_regret_improvement'])} | {r['top8_selection_overlap_count']}/8 | {fmt(r['selected_mean_gain_delta'])} | {'PASS' if r['preliminary_gate_pass'] else 'FAIL'} |")
    passes = int(delta_df.preliminary_gate_pass.sum())
    if len(external_names) == 1:
        verdict = "单序列外部试验完成，但不足以支持广泛泛化结论。即使门槛通过，也只记为单序列初步支持。"
    elif passes >= 2 and passes == len(external_names):
        verdict = "所有外部序列均通过预先固定的初步门槛；可考虑进入下一阶段的同实际预算在线策略验证，但仍需谨慎解释样本相关性。"
    else:
        verdict = "主候选未在所有外部序列上同时通过初步门槛；暂缓在线策略开发，不得根据外部结果调参后重新宣称独立验证。"
    lines += ["", "## 5. 判定", "", verdict, "",
              "初步门槛要求主候选相对年龄基线同时改善 MAE、Spearman，Top 8 regret 不恶化，且 Top 8 选中样本的实际平均收益不降低。该门槛是工程筛选规则，不是统计显著性检验。", "",
              "## 6. 限制", "",
              "- synchronous 曾用于 v5.10/v5.11；本轮按锁定模型复核解释，不视为全项目全新独立测试。",
              "- H=16 窗口跨锚点重叠，样本不独立；不据此报告普通独立样本显著性。",
              "- 本轮只有一条复核序列，不能支撑广泛泛化结论。",
              "- v5.15 标签执行了分支自动重启，但不重放未来主动刷新动作，因此不等同于完整在线策略收益。",
              "- Top 8 只是在已观察锚点中选 K=8 的离线诊断；没有执行在线动作，也没有测量真实计算成本。",
              "- 不修改 VeloDepth 权重，不训练主模型。", ""]
    (out / "P2_3_v5_21_实验报告.md").write_text("\n".join(lines), encoding="utf-8")
    # Build a review archive outside the results directory so it never includes itself.
    from zipfile import ZipFile, ZIP_DEFLATED
    zip_path = script_dir / "P2_3_v5_21_external_validation_review.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file(): zf.write(p, p.relative_to(script_dir) if p.is_relative_to(script_dir) else Path("results") / p.name)
        if train_path.is_file(): zf.write(train_path, "inputs/p2_3_v5_18_aligned_53_anchor_dataset.csv")
        for name in ["analyze_p2_3_v5_21_external_validation.py", "p2_3_v5_21_external_sequence_collector.py", "README_P2_3_v5_21.md", "P2_3_v5_21_实验设计与判据.md"]:
            p = script_dir / name
            if p.is_file(): zf.write(p, name)
    print(f"PASS: v5.21 external evaluation completed for {len(external_names)} unseen sequence(s).")
    print(f"Training rows: {len(train)}; external anchors: {sum(int(a['valid_rows_used']) for a in audits)}")
    print(f"Report: {out / 'P2_3_v5_21_实验报告.md'}")
    print(f"Review ZIP: {zip_path}")

if __name__ == "__main__":
    main()
