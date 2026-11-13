#!/usr/bin/env python3
"""Pure NumPy helpers for R3.3 candidate geometry decomposition and gain analysis."""
from __future__ import annotations
import math
import numpy as np


def _squeeze_depth(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D depth map after squeeze, got shape={arr.shape}")
    return arr


def _robust_affine_fit(x, y, max_fit_samples=50000, iterations=6):
    """Fit y ~= scale*x + shift with deterministic Huber IRLS."""
    n = int(x.size)
    if n < 8:
        return float("nan"), float("nan")
    if n > max_fit_samples:
        step = int(math.ceil(n / max_fit_samples))
        xs, ys = x[::step], y[::step]
    else:
        xs, ys = x, y
    A = np.column_stack((xs, np.ones_like(xs)))
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    for _ in range(iterations):
        residual = ys - A @ beta
        center = np.median(residual)
        mad = np.median(np.abs(residual - center))
        scale = max(1.4826 * float(mad), 1e-8)
        u = np.abs(residual - center) / (1.345 * scale)
        weights = np.ones_like(u)
        large = u > 1.0
        weights[large] = 1.0 / u[large]
        aw = A * np.sqrt(weights)[:, None]
        yw = ys * np.sqrt(weights)
        try:
            beta, *_ = np.linalg.lstsq(aw, yw, rcond=None)
        except np.linalg.LinAlgError:
            break
    return float(beta[0]), float(beta[1])


def decompose_depth_difference(candidate_depth, propagation_depth):
    """Compute raw, global scale/shift, and local residual geometry descriptors.

    No GT is used. Positive finite depth pixels shared by both maps define the mask.
    The fit is candidate ~= scale * propagation + shift.
    """
    c = _squeeze_depth(candidate_depth)
    p = _squeeze_depth(propagation_depth)
    if c.shape != p.shape:
        raise ValueError(f"Depth shape mismatch: candidate={c.shape}, propagation={p.shape}")
    valid = np.isfinite(c) & np.isfinite(p) & (c > 0) & (p > 0)
    n = int(valid.sum())
    out = {
        "geometry_valid_pixels": n,
        "geometry_valid_fraction": float(n / valid.size) if valid.size else 0.0,
    }
    if n < 8:
        keys = [
            "raw_rel_mean", "raw_rel_median", "raw_rel_p90", "scale_shift_scale",
            "scale_shift_shift", "scale_shift_shift_over_median_prop",
            "scale_shift_abs_scale_deviation", "fit_residual_norm_mean",
            "fit_residual_norm_median", "fit_residual_norm_p90",
            "fit_residual_fraction_gt_0p05", "fit_residual_grid_cv", "fit_residual_grid_max",
        ]
        out.update({k: float("nan") for k in keys})
        return out
    cv = c[valid]
    pv = p[valid]
    raw_rel = np.abs(cv - pv) / np.maximum(np.abs(pv), 1e-8)
    a, b = _robust_affine_fit(pv, cv)
    fitted = a * pv + b
    norm_scale = max(float(np.median(np.abs(pv))), 1e-8)
    resid_norm = np.abs(cv - fitted) / norm_scale

    out.update({
        "raw_rel_mean": float(np.mean(raw_rel)),
        "raw_rel_median": float(np.median(raw_rel)),
        "raw_rel_p90": float(np.quantile(raw_rel, 0.90)),
        "scale_shift_scale": a,
        "scale_shift_shift": b,
        "scale_shift_shift_over_median_prop": float(b / norm_scale),
        "scale_shift_abs_scale_deviation": float(abs(a - 1.0)),
        "fit_residual_norm_mean": float(np.mean(resid_norm)),
        "fit_residual_norm_median": float(np.median(resid_norm)),
        "fit_residual_norm_p90": float(np.quantile(resid_norm, 0.90)),
        "fit_residual_fraction_gt_0p05": float(np.mean(resid_norm > 0.05)),
    })

    # 4x4 spatial grid: local residual summaries are normalized by median propagation depth.
    resid_map = np.full(c.shape, np.nan, dtype=np.float64)
    resid_map[valid] = np.abs(c[valid] - (a * p[valid] + b)) / norm_scale
    grid_means = []
    h, w = c.shape
    for yi in range(4):
        y0, y1 = round(yi * h / 4), round((yi + 1) * h / 4)
        for xi in range(4):
            x0, x1 = round(xi * w / 4), round((xi + 1) * w / 4)
            tile = resid_map[y0:y1, x0:x1]
            finite = tile[np.isfinite(tile)]
            if finite.size >= 8:
                grid_means.append(float(np.mean(finite)))
    if grid_means:
        gm = np.asarray(grid_means, dtype=np.float64)
        out["fit_residual_grid_cv"] = float(np.std(gm) / max(abs(float(np.mean(gm))), 1e-8))
        out["fit_residual_grid_max"] = float(np.max(gm))
    else:
        out["fit_residual_grid_cv"] = float("nan")
        out["fit_residual_grid_max"] = float("nan")
    return out


def discounted_gain(rows_b0, rows_b1, gamma=0.9):
    """Return discount-sum and per-valid-pair gain; missing pairs keep original k exponent."""
    total = 0.0
    pairs = 0
    for k, (m0, m1) in enumerate(zip(rows_b0, rows_b1), 1):
        try:
            a0, a1 = float(m0.get("absrel", float("nan"))), float(m1.get("absrel", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(a0) and math.isfinite(a1):
            total += (gamma ** k) * (a0 - a1)
            pairs += 1
    return total, (total / pairs if pairs else float("nan")), pairs


def diagnostic_anchors(frames, offset=1, stride=4, horizon=16):
    return [t for t in range(offset, frames) if (t - offset) % stride == 0 and t + horizon < frames]


def _rankdata_average_ties(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _corr(x, y):
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def association_rows(rows, feature_names, target="gain_absrel_discounted_per_valid_frame"):
    """Descriptive feature/gain associations; no p-values or threshold selection."""
    output = []
    scopes = [("all", None)]
    for seq in sorted({str(r.get("sequence", "")) for r in rows}):
        scopes.append(("sequence", seq))
    for scope, seq in scopes:
        subset = [r for r in rows if (seq is None or str(r.get("sequence", "")) == seq)
                   and str(r.get("counterfactual_status", r.get("status", "ok"))) in ("ok", "completed")]
        for feature in feature_names:
            xs, ys = [], []
            for r in subset:
                try:
                    x, y = float(r.get(feature, float("nan"))), float(r.get(target, float("nan")))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    xs.append(x); ys.append(y)
            x, y = np.asarray(xs), np.asarray(ys)
            if len(x):
                ranks_x, ranks_y = _rankdata_average_ties(x), _rankdata_average_ties(y)
                q25, q75 = np.quantile(x, [0.25, 0.75])
                low = y[x <= q25]
                high = y[x >= q75]
                mean_low = float(np.mean(low)) if low.size else float("nan")
                mean_high = float(np.mean(high)) if high.size else float("nan")
                positive_rate = float(np.mean(y > 0))
            else:
                mean_low = mean_high = positive_rate = float("nan")
                ranks_x = ranks_y = np.asarray([], dtype=float)
            output.append({
                "scope": scope, "sequence": seq or "all", "feature": feature, "n": int(len(x)),
                "pearson_r_descriptive": _corr(x, y) if len(x) else float("nan"),
                "spearman_r_descriptive": _corr(ranks_x, ranks_y) if len(x) else float("nan"),
                "positive_gain_rate": positive_rate,
                "mean_gain_low_feature_quartile": mean_low,
                "mean_gain_high_feature_quartile": mean_high,
                "high_minus_low_quartile_gain": mean_high - mean_low if math.isfinite(mean_high) and math.isfinite(mean_low) else float("nan"),
                "interpretation_limit": "descriptive only; overlapping future windows are dependent; no threshold selected",
            })
    return output
