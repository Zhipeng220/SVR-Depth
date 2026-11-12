#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.3 v5.22: sequence-shift and decision-gain diagnostic (CPU only).

Purpose:
  1) Reproduce locked 3-fold leave-one-sequence-out (LOSO) diagnostics on the
     existing 53-anchor dataset without tuning alpha/features.
  2) Decompose the synchronous v5.21 prediction bias into the age baseline and
     the residualized state-channel-2 correction.
  3) Quantify feature-residual distribution shifts and compare fixed Top-8
     decisions against the state-age baseline.

No VeloDepth inference is run. No VeloDepth weights are changed. The Ridge
regressors here are lightweight diagnostic models, not the VeloDepth model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

VERSION = "P2.3 v5.22"
TARGET = "G_state_absrel_restart_aware_discounted_per_valid_frame"
AGE = "scheduler_distance_kf"
FRAME = "anchor_frame"
STATE2 = "state_features_teacher_2_mean"
RADIUS = "state_radius_teacher_mean"
EXPECTED_TRAIN_SEQUENCES = {"balloon2", "crowd2", "crowd3"}
EXPECTED_TRAIN_ROWS = 53
LOCKED_ALPHA = 10.0
LOCKED_BUDGET_K = 8
EXTERNAL_SEQUENCE = "synchronous"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def finite_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def finite_std(values: Any, ddof: int = 1) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=ddof)) if len(arr) > ddof else float("nan")


def corr(a: Any, b: Any, method: str = "spearman") -> float:
    x = pd.to_numeric(pd.Series(a), errors="coerce")
    y = pd.to_numeric(pd.Series(b), errors="coerce")
    mask = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    x, y = x[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method=method))


def repo_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "experiments" / "p1").is_dir():
            raise FileNotFoundError(f"Not a VeloDepth project root: {root}")
        return root
    env_root = os.environ.get("SVRDEPTH_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "experiments" / "p1").is_dir():
            return root
    script = Path(__file__).resolve()
    for candidate in script.parents:
        if (candidate / "experiments" / "p1").is_dir() and (candidate / "velodepth").is_dir():
            return candidate
    if len(script.parents) >= 3:
        candidate = script.parents[2]
        if (candidate / "experiments" / "p1").is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot locate repository root. Run this script from the VeloDepth project, "
        "or pass --repo-root /root/autodl-tmp/SVR-Depth/velodepth."
    )


def first_existing(paths: list[Path], description: str) -> Path:
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            return path.resolve()
    checked = "\n".join(f"  - {p}" for p in paths)
    raise FileNotFoundError(f"Cannot find {description}. Checked:\n{checked}")


def resolve_inputs(root: Path, train_arg: str | None, external_arg: str | None,
                   out_arg: str | None) -> tuple[Path, Path, Path]:
    p1 = root / "experiments" / "p1"
    if train_arg:
        train_path = Path(train_arg).expanduser().resolve()
    else:
        train_path = first_existing([
            p1 / "P2_3_v5_20_package" / "inputs" / "p2_3_v5_18_aligned_53_anchor_dataset.csv",
            root / "outputs" / "p2_3_v5_21_external" / "synchronous" / "evaluation" / "input_snapshots" / "locked_training_dataset.csv",
            root / "outputs" / "p2_3_v5_21_external" / "synchronous" / "evaluation" / "locked_training_dataset.csv",
        ], "locked 53-anchor training CSV")
    if external_arg:
        external_path = Path(external_arg).expanduser().resolve()
    else:
        external_path = first_existing([
            root / "outputs" / "p2_3_v5_21_external" / "synchronous" / "evaluation" / "synchronous_aligned_external_anchors.csv",
            root / "outputs" / "p2_3_v5_21_external" / "synchronous" / "evaluation" / "synchronous_aligned_external_anchors.csv",
            root / "outputs" / "p2_3_v5_21_external" / "synchronous" / "p2_3_v5_15_same_state_anchor_labels.csv",
            p1 / "P2_3_v5_21_runtime" / "results" / "synchronous_aligned_external_anchors.csv",
        ], "v5.21 synchronous aligned-anchor CSV or raw label CSV")
    out_dir = Path(out_arg).expanduser().resolve() if out_arg else root / "outputs" / "p2_3_v5_22_sequence_shift_diagnostic"
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV does not exist: {train_path}")
    if not external_path.is_file():
        raise FileNotFoundError(f"External CSV does not exist: {external_path}")
    return train_path, external_path, out_dir


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        AGE: "feature_scheduler_distance_kf",
        STATE2: "feature_state_features_teacher_2_mean",
        RADIUS: "feature_state_radius_teacher_mean",
    }
    for canonical, alias in aliases.items():
        if canonical not in out.columns and alias in out.columns:
            out[canonical] = out[alias]
    return out


def load_train(path: Path) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))
    required = ["sequence", "trajectory_policy", "horizon_H", "status", FRAME, AGE, STATE2, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV missing required columns: {missing}")
    df["sequence"] = df["sequence"].astype(str).str.strip().str.lower()
    df["trajectory_policy"] = df["trajectory_policy"].astype(str).str.strip().str.lower()
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df = df[(df.trajectory_policy == "auto") & (to_num(df.horizon_H) == 16) & (df.status == "ok")].copy()
    for col in [FRAME, AGE, STATE2, TARGET] + ([RADIUS] if RADIUS in df.columns else []):
        df[col] = to_num(df[col])
    df = df[np.isfinite(df[[FRAME, AGE, STATE2, TARGET]].to_numpy(float)).all(axis=1)].copy()
    if df.duplicated(["sequence", FRAME]).any():
        raise ValueError("Training data has duplicate sequence+anchor rows")
    seqs = set(df.sequence)
    if seqs != EXPECTED_TRAIN_SEQUENCES:
        raise ValueError(f"Expected exactly {sorted(EXPECTED_TRAIN_SEQUENCES)}; found {sorted(seqs)}")
    if len(df) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Expected exactly {EXPECTED_TRAIN_ROWS} valid training anchors; got {len(df)}")
    return df.sort_values(["sequence", FRAME]).reset_index(drop=True)


def load_external(path: Path) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))
    required = ["sequence", "trajectory_policy", "horizon_H", "status", FRAME, AGE, STATE2, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"External CSV missing required columns: {missing}\nFile: {path}")
    df["sequence"] = df["sequence"].astype(str).str.strip().str.lower()
    df["trajectory_policy"] = df["trajectory_policy"].astype(str).str.strip().str.lower()
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df = df[(df.trajectory_policy == "auto") & (to_num(df.horizon_H) == 16) & (df.status == "ok")].copy()
    for col in [FRAME, AGE, STATE2, TARGET] + ([RADIUS] if RADIUS in df.columns else []):
        df[col] = to_num(df[col])
    df = df[np.isfinite(df[[FRAME, AGE, STATE2, TARGET]].to_numpy(float)).all(axis=1)].copy()
    seqs = set(df.sequence)
    if seqs != {EXTERNAL_SEQUENCE}:
        raise ValueError(f"v5.22 is locked to the existing v5.21 sequence '{EXTERNAL_SEQUENCE}'; found {sorted(seqs)}")
    if len(df) < 12:
        raise ValueError(f"Only {len(df)} valid synchronous auto/H=16 anchors; expected at least 12")
    if df.duplicated(["sequence", FRAME]).any():
        raise ValueError("External data has duplicate sequence+anchor rows")
    return df.sort_values(["sequence", FRAME]).reset_index(drop=True)


def fit_locked_model(train: pd.DataFrame) -> dict[str, Any]:
    y = train[TARGET].to_numpy(float)
    age = train[[AGE]].to_numpy(float)
    x = train[STATE2].to_numpy(float)
    median = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
    x = np.where(np.isfinite(x), x, median)
    age_model = LinearRegression().fit(age, y)
    y_base_train = age_model.predict(age)
    x_age_model = LinearRegression().fit(age, x)
    x_res_train = x - x_age_model.predict(age)
    scaler = StandardScaler().fit(x_res_train.reshape(-1, 1))
    z_train = scaler.transform(x_res_train.reshape(-1, 1))
    y_res_train = y - y_base_train
    ridge = Ridge(alpha=LOCKED_ALPHA).fit(z_train, y_res_train)
    return {
        "age_model": age_model,
        "x_age_model": x_age_model,
        "scaler": scaler,
        "ridge": ridge,
        "feature_median": median,
        "train_x_residual": x_res_train,
        "train_z": z_train.ravel(),
        "train_y_residual": y_res_train,
        "train_base_prediction": y_base_train,
    }


def predict_locked(model: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    age = df[[AGE]].to_numpy(float)
    raw = df[STATE2].to_numpy(float)
    raw = np.where(np.isfinite(raw), raw, model["feature_median"])
    residual = raw - model["x_age_model"].predict(age)
    z = model["scaler"].transform(residual.reshape(-1, 1)).ravel()
    base = model["age_model"].predict(age)
    correction = model["ridge"].predict(z.reshape(-1, 1))
    return pd.DataFrame({
        "feature_residual": residual,
        "feature_residual_z": z,
        "age_baseline_prediction": base,
        "state2_residual_correction": correction,
        "candidate_prediction": base + correction,
    })


def evaluate(y: np.ndarray, pred: np.ndarray, budget_k: int = LOCKED_BUDGET_K) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) != len(pred) or len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Invalid y/pred arrays passed to evaluate")
    n = len(y)
    k = min(budget_k, n)
    chosen = np.argsort(-pred, kind="stable")[:k]
    oracle = np.argsort(-y, kind="stable")[:k]
    return {
        "n": int(n),
        "target_mean": float(np.mean(y)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_bias": float(np.mean(pred - y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if n > 1 else float("nan"),
        "spearman": corr(pred, y),
        "top8_k": int(k),
        "top8_selected_mean_gain": float(np.mean(y[chosen])),
        "top8_oracle_mean_gain": float(np.mean(y[oracle])),
        "top8_regret": float(np.mean(y[oracle]) - np.mean(y[chosen])),
        "top8_lift": float(np.mean(y[chosen]) - np.mean(y)),
        "top8_indices": chosen.tolist(),
        "oracle_indices": oracle.tolist(),
    }


def distribution_rows(train: pd.DataFrame, external: pd.DataFrame, full_model: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split, frame in [("train", train), ("external_v5_21", external)]:
        pred_parts = predict_locked(full_model, frame)
        for seq, group in frame.groupby("sequence", sort=True):
            ix = group.index.to_numpy()
            part = pred_parts.loc[ix]
            y = group[TARGET].to_numpy(float)
            age = group[AGE].to_numpy(float)
            x = group[STATE2].to_numpy(float)
            y_res = y - full_model["age_model"].predict(group[[AGE]].to_numpy(float))
            rows.append({
                "split": split,
                "sequence": seq,
                "n": int(len(group)),
                "age_mean": finite_mean(age),
                "age_std": finite_std(age),
                "state2_raw_mean": finite_mean(x),
                "state2_raw_std": finite_std(x),
                "target_mean": finite_mean(y),
                "target_std": finite_std(y),
                "raw_state2_target_spearman": corr(x, y),
                "age_target_spearman": corr(age, y),
                "feature_residual_mean": finite_mean(part["feature_residual"]),
                "feature_residual_std": finite_std(part["feature_residual"]),
                "feature_residual_z_mean": finite_mean(part["feature_residual_z"]),
                "feature_residual_z_std": finite_std(part["feature_residual_z"]),
                "age_residual_target_spearman": corr(part["feature_residual"], y_res),
                "feature_correction_mean": finite_mean(part["state2_residual_correction"]),
            })
    return pd.DataFrame(rows)


def run_loso(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_rows, metric_rows, fold_rows = [], [], []
    for heldout in sorted(train.sequence.unique()):
        tr = train[train.sequence != heldout].reset_index(drop=True)
        te = train[train.sequence == heldout].reset_index(drop=True)
        model = fit_locked_model(tr)
        pred_parts = predict_locked(model, te)
        y = te[TARGET].to_numpy(float)
        base_pred = pred_parts["age_baseline_prediction"].to_numpy(float)
        candidate_pred = pred_parts["candidate_prediction"].to_numpy(float)
        base_m = evaluate(y, base_pred)
        cand_m = evaluate(y, candidate_pred)
        k = min(LOCKED_BUDGET_K, len(te))
        base_idx = set(np.argsort(-base_pred, kind="stable")[:k].tolist())
        cand_idx = set(np.argsort(-candidate_pred, kind="stable")[:k].tolist())
        base_actual_mean = float(np.mean(y[list(base_idx)]))
        cand_actual_mean = float(np.mean(y[list(cand_idx)]))
        delta = {
            "heldout_sequence": heldout,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "train_sequences": ";".join(sorted(tr.sequence.unique())),
            "feature_median_train": model["feature_median"],
            "ridge_coefficient_standardized": float(model["ridge"].coef_[0]),
            "train_feature_residual_mean": finite_mean(model["train_x_residual"]),
            "train_feature_residual_std": finite_std(model["train_x_residual"], ddof=0),
            "test_feature_residual_mean": finite_mean(pred_parts["feature_residual"]),
            "test_feature_residual_std": finite_std(pred_parts["feature_residual"], ddof=0),
            "test_feature_residual_z_mean": finite_mean(pred_parts["feature_residual_z"]),
            "age_baseline_mae": base_m["mae"],
            "candidate_mae": cand_m["mae"],
            "mae_improvement_base_minus_candidate": base_m["mae"] - cand_m["mae"],
            "age_baseline_spearman": base_m["spearman"],
            "candidate_spearman": cand_m["spearman"],
            "spearman_delta_candidate_minus_base": cand_m["spearman"] - base_m["spearman"],
            "age_baseline_top8_regret": base_m["top8_regret"],
            "candidate_top8_regret": cand_m["top8_regret"],
            "top8_regret_improvement_base_minus_candidate": base_m["top8_regret"] - cand_m["top8_regret"],
            "age_baseline_top8_selected_mean": base_actual_mean,
            "candidate_top8_selected_mean": cand_actual_mean,
            "top8_selected_mean_delta_candidate_minus_base": cand_actual_mean - base_actual_mean,
            "top8_overlap_count": len(base_idx & cand_idx),
            "top8_overlap_rate": len(base_idx & cand_idx) / max(k, 1),
            "candidate_mae_improved": bool(cand_m["mae"] < base_m["mae"] - 1e-12),
            "candidate_decision_improved": bool(
                cand_m["top8_regret"] < base_m["top8_regret"] - 1e-12
                and cand_actual_mean > base_actual_mean + 1e-12
            ),
            "candidate_decision_not_worse": bool(
                cand_m["top8_regret"] <= base_m["top8_regret"] + 1e-12
                and cand_actual_mean >= base_actual_mean - 1e-12
            ),
        }
        fold_rows.append(delta)
        for model_name, pred, metrics in [
            ("state_age_ols", base_pred, base_m),
            ("age_resid_state2_mean", candidate_pred, cand_m),
        ]:
            chosen = metrics["top8_indices"]
            selected_set = set(chosen)
            for i, row in te.iterrows():
                pred_rows.append({
                    "heldout_sequence": heldout,
                    "anchor_frame": float(row[FRAME]),
                    "model": model_name,
                    "prediction": float(pred[i]),
                    "target": float(y[i]),
                    "prediction_error": float(pred[i] - y[i]),
                    "selected_top8": bool(i in selected_set),
                    "feature_state2": float(row[STATE2]),
                    "state_age": float(row[AGE]),
                    "feature_residual": float(pred_parts.iloc[i]["feature_residual"]),
                    "feature_residual_z": float(pred_parts.iloc[i]["feature_residual_z"]),
                })
            metric_rows.append({"heldout_sequence": heldout, "model": model_name, **{k: v for k, v in metrics.items() if not isinstance(v, list)}})
    return pd.DataFrame(pred_rows), pd.DataFrame(metric_rows), pd.DataFrame(fold_rows)


def run_external(train: pd.DataFrame, external: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = fit_locked_model(train)
    parts = predict_locked(model, external)
    y = external[TARGET].to_numpy(float)
    base_pred = parts["age_baseline_prediction"].to_numpy(float)
    cand_pred = parts["candidate_prediction"].to_numpy(float)
    base_m = evaluate(y, base_pred)
    cand_m = evaluate(y, cand_pred)
    k = min(LOCKED_BUDGET_K, len(external))
    base_idx = np.argsort(-base_pred, kind="stable")[:k]
    cand_idx = np.argsort(-cand_pred, kind="stable")[:k]
    oracle_idx = np.argsort(-y, kind="stable")[:k]
    frame_values = external[FRAME].to_numpy(float)
    out = external[["sequence", FRAME, AGE, STATE2, TARGET]].copy()
    out = out.rename(columns={TARGET: "target_gain"})
    out["feature_residual"] = parts["feature_residual"].to_numpy(float)
    out["feature_residual_z"] = parts["feature_residual_z"].to_numpy(float)
    out["age_baseline_prediction"] = base_pred
    out["state2_residual_correction"] = parts["state2_residual_correction"].to_numpy(float)
    out["candidate_prediction"] = cand_pred
    out["age_baseline_error"] = base_pred - y
    out["candidate_error"] = cand_pred - y
    out["selected_top8_by_age"] = [i in set(base_idx.tolist()) for i in range(len(external))]
    out["selected_top8_by_candidate"] = [i in set(cand_idx.tolist()) for i in range(len(external))]
    out["oracle_top8"] = [i in set(oracle_idx.tolist()) for i in range(len(external))]
    out["rank_by_age"] = pd.Series(-base_pred).rank(method="first").astype(int).to_numpy()
    out["rank_by_candidate"] = pd.Series(-cand_pred).rank(method="first").astype(int).to_numpy()
    out["rank_by_oracle"] = pd.Series(-y).rank(method="first").astype(int).to_numpy()
    out = out.sort_values(FRAME).reset_index(drop=True)

    metrics = pd.DataFrame([
        {"sequence": EXTERNAL_SEQUENCE, "model": "state_age_ols", **{k: v for k, v in base_m.items() if not isinstance(v, list)}},
        {"sequence": EXTERNAL_SEQUENCE, "model": "age_resid_state2_mean", **{k: v for k, v in cand_m.items() if not isinstance(v, list)}},
    ])
    bias = pd.DataFrame([{
        "sequence": EXTERNAL_SEQUENCE,
        "n": int(len(external)),
        "target_mean": finite_mean(y),
        "target_std": finite_std(y),
        "age_baseline_prediction_mean": finite_mean(base_pred),
        "age_baseline_prediction_std": finite_std(base_pred),
        "age_baseline_bias_pred_minus_target": finite_mean(base_pred - y),
        "age_baseline_mae": base_m["mae"],
        "feature_residual_train_mean": finite_mean(model["train_x_residual"]),
        "feature_residual_train_std": finite_std(model["train_x_residual"], ddof=0),
        "feature_residual_external_mean": finite_mean(parts["feature_residual"]),
        "feature_residual_external_std": finite_std(parts["feature_residual"], ddof=0),
        "feature_residual_z_external_mean": finite_mean(parts["feature_residual_z"]),
        "feature_residual_z_external_std": finite_std(parts["feature_residual_z"], ddof=0),
        "candidate_correction_mean": finite_mean(parts["state2_residual_correction"]),
        "candidate_correction_std": finite_std(parts["state2_residual_correction"], ddof=0),
        "candidate_prediction_mean": finite_mean(cand_pred),
        "candidate_prediction_std": finite_std(cand_pred),
        "candidate_bias_pred_minus_target": finite_mean(cand_pred - y),
        "candidate_mae": cand_m["mae"],
        "candidate_mae_improvement_base_minus_candidate": base_m["mae"] - cand_m["mae"],
        "candidate_spearman": cand_m["spearman"],
        "age_baseline_spearman": base_m["spearman"],
        "candidate_spearman_delta": cand_m["spearman"] - base_m["spearman"],
        "age_baseline_top8_regret": base_m["top8_regret"],
        "candidate_top8_regret": cand_m["top8_regret"],
        "top8_regret_improvement_base_minus_candidate": base_m["top8_regret"] - cand_m["top8_regret"],
        "age_baseline_top8_selected_mean": base_m["top8_selected_mean_gain"],
        "candidate_top8_selected_mean": cand_m["top8_selected_mean_gain"],
        "top8_selected_mean_delta": cand_m["top8_selected_mean_gain"] - base_m["top8_selected_mean_gain"],
        "top8_overlap_count": int(len(set(base_idx.tolist()) & set(cand_idx.tolist()))),
        "ridge_alpha_locked": LOCKED_ALPHA,
        "ridge_coefficient_standardized": float(model["ridge"].coef_[0]),
        "ridge_intercept_residual_target": float(model["ridge"].intercept_),
    }])
    return out, metrics, bias


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, digits: int = 5) -> str:
    view = df[columns].copy() if columns else df.copy()
    headers = list(view.columns)
    def fmt(v: Any) -> str:
        if pd.isna(v):
            return "NA"
        if isinstance(v, (bool, np.bool_)):
            return "是" if bool(v) else "否"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{digits}f}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    rows = [[fmt(v) for v in row] for row in view.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_report(out_dir: Path, train: pd.DataFrame, external: pd.DataFrame,
                 distributions: pd.DataFrame, loso_metrics: pd.DataFrame,
                 loso_folds: pd.DataFrame, external_metrics: pd.DataFrame,
                 external_bias: pd.DataFrame, train_path: Path, external_path: Path) -> str:
    ext = external_bias.iloc[0]
    fold_count = len(loso_folds)
    mae_improved_count = int(loso_folds["candidate_mae_improved"].sum())
    decision_improved_count = int(loso_folds["candidate_decision_improved"].sum())
    decision_not_worse_count = int(loso_folds["candidate_decision_not_worse"].sum())
    if ext["candidate_mae_improvement_base_minus_candidate"] > 0 and (
        ext["top8_regret_improvement_base_minus_candidate"] < -1e-12
        or ext["top8_selected_mean_delta"] < -1e-12
    ):
        external_decision = "MAE 改善与 Top 8 决策质量脱钩；固定预算决策未通过。"
    elif ext["candidate_mae_improvement_base_minus_candidate"] <= 0:
        external_decision = "主候选的 MAE 未优于状态年龄基线；当前证据不支持继续将其作为主候选。"
    else:
        external_decision = "本次外部序列的决策指标没有出现上述明显冲突；仍须受单序列与历史暴露限制。"
    correction_dominates = (
        abs(ext["candidate_bias_pred_minus_target"]) > abs(ext["age_baseline_bias_pred_minus_target"])
        and abs(ext["candidate_correction_mean"]) > 1e-12
    )
    lines = [
        "# P2.3 v5.22 实验报告：跨序列偏移与决策增益诊断",
        "",
        "## 1. 实验定位",
        "本轮复用 v5.21 锁定的 53 个训练锚点与 synchronous H=16 标签，只做 CPU 离线诊断。",
        "不执行 VeloDepth 推理、不修改或训练 VeloDepth 权重、不搜索新特征、不调整 Ridge alpha=10，也不将本轮解释为新的独立外部验证。",
        "Ridge 仅作为轻量收益诊断器在 LOSO 折内拟合，不是 VeloDepth 主模型。",
        "",
        "## 2. 数据核验",
        f"- 训练样本：{len(train)}；训练序列：{', '.join(sorted(train.sequence.unique()))}。",
        f"- v5.21 复核序列：{external.sequence.iloc[0]}；有效锚点：{len(external)}。",
        f"- 训练 CSV：`{train_path}`。",
        f"- 外部 CSV：`{external_path}`。",
        "- 固定配置：目标为重启感知 H=16 同状态收益；状态年龄基线 OLS；状态通道 2 年龄残差化 + StandardScaler + Ridge(alpha=10)；离线预算 K=8。",
        "- 注意：synchronous 曾用于 v5.10/v5.11；本轮是锁定模型复核，不是项目历史上从未接触过的新序列。",
        "",
        "## 3. 三折留一序列诊断（训练集内部，仅探索性）",
        f"- 候选 MAE 优于年龄基线：{mae_improved_count}/{fold_count} 折。",
        f"- 候选 Top 8 regret 与选中平均收益同时严格改善：{decision_improved_count}/{fold_count} 折。",
        f"- 候选 Top 8 决策没有恶化（regret 不升且选中均值不降）：{decision_not_worse_count}/{fold_count} 折。",
        "- 每折完整留出一条序列；填补、年龄回归、特征残差化、标准化和 Ridge 均仅使用另外两条序列拟合。",
        "",
        markdown_table(loso_folds, [
            "heldout_sequence", "n_train", "n_test", "ridge_coefficient_standardized",
            "test_feature_residual_z_mean", "age_baseline_mae", "candidate_mae",
            "mae_improvement_base_minus_candidate", "age_baseline_top8_regret", "candidate_top8_regret",
            "top8_regret_improvement_base_minus_candidate", "top8_selected_mean_delta_candidate_minus_base",
            "top8_overlap_count", "candidate_mae_improved", "candidate_decision_improved", "candidate_decision_not_worse",
        ]),
        "",
        "## 4. 训练序列与复核序列的特征偏移",
        "下表使用完整 53 锚点训练集拟合的锁定预处理器，将训练序列和 synchronous 映射到同一残差特征尺度。相关系数仅作描述，不是因果或显著性证据。",
        "",
        markdown_table(distributions, [
            "split", "sequence", "n", "age_mean", "state2_raw_mean", "state2_raw_std", "target_mean",
            "raw_state2_target_spearman", "age_target_spearman", "feature_residual_mean",
            "feature_residual_std", "feature_residual_z_mean", "feature_residual_z_std",
            "age_residual_target_spearman", "feature_correction_mean",
        ]),
        "",
        "## 5. synchronous 的预测偏差分解",
        markdown_table(external_bias, [
            "sequence", "n", "target_mean", "age_baseline_prediction_mean", "age_baseline_bias_pred_minus_target",
            "age_baseline_mae", "feature_residual_train_mean", "feature_residual_train_std",
            "feature_residual_external_mean", "feature_residual_external_std", "feature_residual_z_external_mean",
            "feature_residual_z_external_std", "candidate_correction_mean", "candidate_prediction_mean",
            "candidate_bias_pred_minus_target", "candidate_mae", "candidate_mae_improvement_base_minus_candidate",
            "age_baseline_spearman", "candidate_spearman", "candidate_spearman_delta",
            "age_baseline_top8_regret", "candidate_top8_regret", "top8_regret_improvement_base_minus_candidate",
            "age_baseline_top8_selected_mean", "candidate_top8_selected_mean", "top8_selected_mean_delta",
            "top8_overlap_count", "ridge_coefficient_standardized",
        ]),
        "",
        f"**外部固定预算判断：** {external_decision}",
        f"**偏差来源提示：** {'加入状态通道 2 修正后，绝对预测偏差比年龄基线更大，且平均修正不为零；这与残差特征分布偏移导致的校准恶化相一致，但不能单凭本轮证明因果。' if correction_dominates else '加入状态通道 2 后没有满足“绝对预测偏差进一步增大且平均修正非零”这一描述性模式。'}",
        "",
        "## 6. 固定预算 K=8 的具体选择",
        "逐锚点预测、特征残差 z 分数、基线/候选预测与 oracle 排名保存在 `p2_3_v5_22_external_predictions.csv`；留一序列逐样本预测保存在 `p2_3_v5_22_loso_predictions.csv`。Oracle 只作为离线参考，不是可部署策略。",
        "",
        "## 7. 结论与下一步",
        "- 如果 MAE 改善但 Top 8 regret 未改善，不能据此保留该候选作为在线刷新控制器；回归拟合质量与预算排序能力必须分开判断。",
        "- 如果复核序列的标准化残差均值明显偏离训练尺度，且残差修正进一步放大预测偏差，则应优先研究序列级校准/稳健性，而不是堆叠新特征或调参。该现象仅为偏移证据，不代表可直接用测试序列校准模型。",
        "- 不允许利用 synchronous 的标签重新拟合或校准模型后，再把同一序列结果称为独立验证。",
        "- 当前仍不进入在线策略扩展或 GPU 实验；下一步先根据本报告决定是停止状态通道 2 候选，还是另立严格的训练内校准研究。",
        "",
        "## 8. 限制",
        "- 训练集仅 3 条序列、53 个锚点；留一序列结果属于探索性诊断。",
        "- H=16 未来窗口可能重叠，锚点并非独立样本；不做普通独立样本显著性检验。",
        "- v5.15 标签执行了分支自动重启，但不重放未来主动刷新动作；Top 8 是离线排序，不等于完整在线策略收益或同实际计算成本对比。",
        "- synchronous 有历史实验暴露，因此不是全项目范围内全新独立测试。",
        "- VeloDepth 权重保持冻结；本轮没有任何 GPU 需求。",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "P2_3_v5_22_实验报告.md").write_text(report, encoding="utf-8")
    return report


def self_test() -> None:
    rng = np.random.default_rng(522)
    age = np.tile(np.arange(1, 13, dtype=float), 2)
    x = 0.01 * age + rng.normal(0, 0.1, size=len(age))
    y = 0.002 * age + 0.02 * (x - 0.01 * age) + rng.normal(0, 0.001, size=len(age))
    synthetic = pd.DataFrame({
        "sequence": ["s1"] * 12 + ["s2"] * 12,
        "trajectory_policy": ["auto"] * 24,
        "horizon_H": [16] * 24,
        "status": ["ok"] * 24,
        FRAME: np.tile(np.arange(1, 13, dtype=float), 2),
        AGE: age,
        STATE2: x,
        TARGET: y,
    })
    model = fit_locked_model(synthetic)
    predicted = predict_locked(model, synthetic)
    assert len(predicted) == len(synthetic)
    assert np.isfinite(predicted.to_numpy(float)).all()
    met = evaluate(y, predicted["candidate_prediction"].to_numpy(float), budget_k=8)
    assert met["top8_k"] == 8
    assert 0.0 <= met["top8_regret"] + 1e-9
    print("PASS: v5.22 synthetic self-tests (locked residual model, finite predictions, Top-8 metrics).")


def run(args: argparse.Namespace) -> int:
    if args.self_test_only:
        self_test()
        return 0
    root = repo_root(args.repo_root)
    train_path, external_path, out_dir = resolve_inputs(root, args.train_csv, args.external_csv, args.out_dir)
    print(f"[{VERSION}] CPU-only sequence-shift diagnostic", flush=True)
    print(f"Training CSV: {train_path}", flush=True)
    print(f"External CSV: {external_path}", flush=True)
    print("Model weights: frozen; no VeloDepth inference; Ridge alpha=10 and K=8 locked.", flush=True)
    train = load_train(train_path)
    external = load_external(external_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = out_dir / "input_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_path, snap_dir / "locked_training_dataset.csv")
    shutil.copy2(external_path, snap_dir / "synchronous_v5_21_external_input.csv")

    distributions = distribution_rows(train, external, fit_locked_model(train))
    loso_predictions, loso_metrics, loso_folds = run_loso(train)
    external_predictions, external_metrics, external_bias = run_external(train, external)

    outputs = {
        "p2_3_v5_22_sequence_distribution.csv": distributions,
        "p2_3_v5_22_loso_predictions.csv": loso_predictions,
        "p2_3_v5_22_loso_sequence_metrics.csv": loso_metrics,
        "p2_3_v5_22_loso_fold_diagnostic.csv": loso_folds,
        "p2_3_v5_22_external_predictions.csv": external_predictions,
        "p2_3_v5_22_external_metrics.csv": external_metrics,
        "p2_3_v5_22_external_bias_decomposition.csv": external_bias,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig")

    model = fit_locked_model(train)
    metadata = {
        "experiment": VERSION + " sequence-shift and decision-gain diagnostic",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_csv": str(train_path),
        "training_csv_sha256": sha256(train_path),
        "training_rows": int(len(train)),
        "training_sequences": sorted(train.sequence.unique().tolist()),
        "external_csv": str(external_path),
        "external_csv_sha256": sha256(external_path),
        "external_sequence": EXTERNAL_SEQUENCE,
        "external_rows": int(len(external)),
        "target": TARGET,
        "age_feature": AGE,
        "locked_candidate_feature": STATE2,
        "ridge_alpha": LOCKED_ALPHA,
        "budget_k": LOCKED_BUDGET_K,
        "validation": "3-fold leave-one-sequence-out on existing 53 anchors; synchronous locked-model replication",
        "external_sequence_historical_exposure": "synchronous used in earlier P2.3 v5.10/v5.11; not pristine project-wide unseen sequence",
        "v5_depth_weights_modified": False,
        "gpu_required": False,
        "feature_residual_train_mean": finite_mean(model["train_x_residual"]),
        "feature_residual_train_std_population": finite_std(model["train_x_residual"], ddof=0),
        "ridge_coefficient_standardized": float(model["ridge"].coef_[0]),
        "limitations": [
            "Exploratory only; three training sequences and 53 anchors.",
            "H=16 windows overlap; no ordinary independent-sample significance tests.",
            "Top-8 is offline ranking, not a full online-policy or equal-measured-compute test.",
            "The target is restart-aware same-state writeback gain, not full proactive-policy gain.",
            "Synchronous has historical P2.3 exposure and is not a pristine project-wide unseen sequence.",
        ],
        "outputs": list(outputs.keys()) + ["P2_3_v5_22_实验报告.md"],
    }
    (out_dir / "p2_3_v5_22_metadata.json").write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(out_dir, train, external, distributions, loso_metrics, loso_folds,
                 external_metrics, external_bias, train_path, external_path)

    zip_path = out_dir.parent / "p2_3_v5_22_sequence_shift_diagnostic_review.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir.parent))
    print("\n===== P2.3 v5.22 quick summary =====")
    print(loso_folds[["heldout_sequence", "mae_improvement_base_minus_candidate",
                      "top8_regret_improvement_base_minus_candidate",
                      "top8_selected_mean_delta_candidate_minus_base", "top8_overlap_count"]].to_string(index=False))
    print("\nExternal bias decomposition:")
    print(external_bias[["target_mean", "age_baseline_prediction_mean", "candidate_correction_mean",
                         "candidate_prediction_mean", "age_baseline_mae", "candidate_mae",
                         "age_baseline_top8_regret", "candidate_top8_regret"]].to_string(index=False))
    print(f"\nReport: {out_dir / 'P2_3_v5_22_实验报告.md'}")
    print(f"Review ZIP: {zip_path}")
    print("PASS: v5.22 diagnostic completed. No GPU inference or VeloDepth weight changes.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--train-csv", default=None, help="Optional explicit path to the locked 53-anchor CSV")
    parser.add_argument("--external-csv", default=None, help="Optional explicit path to the v5.21 synchronous CSV")
    parser.add_argument("--out-dir", default=None, help="Optional output directory")
    parser.add_argument("--self-test-only", action="store_true", help="Run synthetic self-tests only")
    args = parser.parse_args()
    try:
        return run(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
