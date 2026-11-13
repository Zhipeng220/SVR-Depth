#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVR-Depth P2.2.3: fixed-budget offline ranking evaluation.

Uses out-of-sequence predictions produced by P2.2.2; no GPU/model inference.
For each held-out sequence and horizon, rank candidate anchors using model
predictions and select the top 25%, 50%, or 75%. Compare with random,
evenly-spaced temporal selection, no-refresh, and oracle top-k.

Run:
python experiments/p1/p2_2_3_fixed_budget_ranking.py \
  --input-dir outputs/p2_2_2 \
  --out-dir outputs/p2_2_3 \
  --random-trials 1000

Interpretation:
- Targets are P1.5 full-frame state-writeback gains under forced propagation.
- This is offline ranking evaluation, NOT an online B0/B1/B2 experiment.
- All learned predictions must come from held-out-sequence P2.2.2 output.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEQUENCES = ("crowd2", "balloon2", "crowd3")
HORIZONS = (1, 4, 8, 16)
PRIMARY_TARGET = "G_state_absrel_discounted_per_valid_frame"
TARGET_VARIANTS = (
    PRIMARY_TARGET,
    "target_discount_weight_normalized",
)
PREDICTION_MODELS = (
    "motion_only",
    "learned_state_only",
    "motion_plus_learned",
    "geometry_confidence_only",
    "motion_plus_geom_conf",
    "state_motion_full",
)
BUDGETS = (0.25, 0.50, 0.75)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input file missing: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def top_k_indices(values, k, largest=True):
    values = np.asarray(values, dtype=float)
    # Stable tie handling makes results deterministic.
    order = np.argsort(-values if largest else values, kind="mergesort")
    return order[:k]


def temporal_spread_indices(anchors, k):
    """Select k approximately evenly-spaced anchor positions."""
    n = len(anchors)
    if k >= n:
        return np.arange(n, dtype=int)
    raw = np.linspace(0, n - 1, num=k)
    idx = np.rint(raw).astype(int)
    # Rounding can duplicate indices for some n/k combinations.
    idx = list(dict.fromkeys(idx.tolist()))
    if len(idx) < k:
        for candidate in range(n):
            if candidate not in idx:
                idx.append(candidate)
            if len(idx) == k:
                break
    return np.asarray(sorted(idx[:k]), dtype=int)


def score_selection(y, selected_idx, oracle_idx):
    y = np.asarray(y, dtype=float)
    chosen = np.asarray(selected_idx, dtype=int)
    oracle = np.asarray(oracle_idx, dtype=int)
    chosen_y = y[chosen]
    oracle_total = float(np.sum(y[oracle]))
    total = float(np.sum(chosen_y))
    return {
        "selected_n": int(len(chosen)),
        "selected_total_gain": total,
        "selected_mean_gain": float(np.mean(chosen_y)) if len(chosen_y) else np.nan,
        "selected_positive_fraction": float(np.mean(chosen_y > 0)) if len(chosen_y) else np.nan,
        "selected_negative_count": int(np.sum(chosen_y <= 0)),
        "oracle_topk_total_gain": oracle_total,
        "regret_vs_oracle_topk": float(oracle_total - total),
        "gain_per_selected_anchor": float(total / len(chosen)) if len(chosen) else np.nan,
        "beats_no_refresh": bool(total > 0.0),
        "no_refresh_total_gain": 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="P2.2.3 fixed-budget ranking evaluation")
    ap.add_argument("--input-dir", default="outputs/p2_2_2")
    ap.add_argument("--out-dir", default="outputs/p2_2_3")
    ap.add_argument("--random-trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--budgets", default="0.25,0.50,0.75")
    ap.add_argument("--target-variants", default="primary,normalized")
    args = ap.parse_args()

    if args.random_trials < 100:
        ap.error("--random-trials should be at least 100 for a useful random baseline")
    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = input_dir / "p2_2_2_predictions.csv"
    predictions = read_csv(pred_path)
    required = {"sequence", "anchor_frame", "horizon_H", "target_variant",
                "model", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{pred_path} missing columns: {sorted(missing)}")

    budgets = sorted(set(float(x.strip()) for x in args.budgets.split(",") if x.strip()))
    if not budgets or any(not (0 < b <= 1) for b in budgets):
        ap.error("--budgets values must be in (0,1]")
    requested_targets = [x.strip() for x in args.target_variants.split(",") if x.strip()]
    target_map = {
        "primary": PRIMARY_TARGET,
        "normalized": "target_discount_weight_normalized",
    }
    try:
        targets = [target_map[x] for x in requested_targets]
    except KeyError as exc:
        ap.error(f"Unknown target variant {exc}; use primary,normalized")

    # Verify these are genuinely held-out sequence predictions, and reduce to
    # one target per (sequence, anchor, horizon, target variant).
    if "split" in predictions.columns:
        predictions = predictions[predictions["split"].astype(str).isin(
            ["held_out_sequence", "pooled_LOSO"]
        )].copy()
    predictions["anchor_frame"] = pd.to_numeric(predictions["anchor_frame"], errors="raise").astype(int)
    predictions["horizon_H"] = pd.to_numeric(predictions["horizon_H"], errors="raise").astype(int)
    predictions["y_true"] = pd.to_numeric(predictions["y_true"], errors="coerce")
    predictions["y_pred"] = pd.to_numeric(predictions["y_pred"], errors="coerce")
    predictions = predictions[predictions["model"].isin(PREDICTION_MODELS)].copy()

    # Duplicate protection: the P2.2.2 file normally has one row per sample/model.
    key = ["sequence", "anchor_frame", "horizon_H", "target_variant", "model"]
    if predictions.duplicated(key).any():
        predictions = predictions.drop_duplicates(key, keep="first")

    result_rows = []
    random_rows = []
    audit_rows = []

    rng = np.random.default_rng(args.seed)
    for target in targets:
        for horizon in HORIZONS:
            for sequence in SEQUENCES:
                sub = predictions[
                    (predictions["target_variant"] == target) &
                    (predictions["horizon_H"] == horizon) &
                    (predictions["sequence"] == sequence)
                ].copy()
                if sub.empty:
                    continue

                # Ensure all model rankings are compared on the same anchor set.
                pivot_pred = sub.pivot_table(
                    index="anchor_frame", columns="model", values="y_pred", aggfunc="first"
                )
                y_by_anchor = sub.groupby("anchor_frame")["y_true"].first()
                anchors = sorted(set(pivot_pred.index) & set(y_by_anchor.index))
                common_models = [m for m in PREDICTION_MODELS if m in pivot_pred.columns]
                if not anchors:
                    continue
                pivot_pred = pivot_pred.loc[anchors]
                y = y_by_anchor.loc[anchors].to_numpy(dtype=float)
                anchor_arr = np.asarray(anchors, dtype=int)

                finite_y = np.isfinite(y)
                finite_preds = np.all(np.isfinite(pivot_pred[common_models].to_numpy(float)), axis=1)
                keep = finite_y & finite_preds
                if not np.all(keep):
                    audit_rows.append({
                        "sequence": sequence, "horizon_H": horizon, "target_variant": target,
                        "candidate_anchors_before_finite_filter": len(anchors),
                        "anchors_after_finite_filter": int(np.sum(keep)),
                        "dropped_nonfinite_rows": int(np.sum(~keep)),
                    })
                y = y[keep]
                anchor_arr = anchor_arr[keep]
                pivot_pred = pivot_pred.loc[anchor_arr]
                n = len(y)
                if n < 3:
                    continue

                for budget in budgets:
                    k = max(1, int(np.ceil(budget * n)))
                    k = min(k, n)
                    oracle_idx = top_k_indices(y, k, largest=True)
                    temporal_idx = temporal_spread_indices(anchor_arr, k)

                    # Random baseline distribution, fixed budget and same anchors.
                    random_totals = []
                    random_means = []
                    for trial in range(args.random_trials):
                        idx = rng.choice(n, size=k, replace=False)
                        sc = score_selection(y, idx, oracle_idx)
                        random_totals.append(sc["selected_total_gain"])
                        random_means.append(sc["selected_mean_gain"])
                    random_totals = np.asarray(random_totals, dtype=float)
                    random_means = np.asarray(random_means, dtype=float)
                    random_summary = {
                        "sequence": sequence, "horizon_H": horizon,
                        "target_variant": target, "budget_fraction": budget,
                        "n_candidates": n, "selected_k": k,
                        "random_trials": args.random_trials,
                        "random_total_gain_mean": float(np.mean(random_totals)),
                        "random_total_gain_std": float(np.std(random_totals, ddof=1)),
                        "random_total_gain_p05": float(np.quantile(random_totals, 0.05)),
                        "random_total_gain_p50": float(np.quantile(random_totals, 0.50)),
                        "random_total_gain_p95": float(np.quantile(random_totals, 0.95)),
                        "random_mean_gain_mean": float(np.mean(random_means)),
                        "random_probability_beats_no_refresh": float(np.mean(random_totals > 0)),
                    }
                    random_rows.append(random_summary)

                    selections = {
                        "temporal_even_spacing": temporal_idx,
                        "oracle_topk": oracle_idx,
                        "no_refresh": np.asarray([], dtype=int),
                    }
                    for model in common_models:
                        selections[f"rank_{model}"] = top_k_indices(
                            pivot_pred[model].to_numpy(float), k, largest=True
                        )

                    for strategy, idx in selections.items():
                        if strategy == "no_refresh":
                            sc = {
                                "selected_n": 0, "selected_total_gain": 0.0,
                                "selected_mean_gain": np.nan,
                                "selected_positive_fraction": np.nan,
                                "selected_negative_count": 0,
                                "oracle_topk_total_gain": float(np.sum(y[oracle_idx])),
                                "regret_vs_oracle_topk": float(np.sum(y[oracle_idx])),
                                "gain_per_selected_anchor": np.nan,
                                "beats_no_refresh": False,
                                "no_refresh_total_gain": 0.0,
                            }
                        else:
                            sc = score_selection(y, idx, oracle_idx)
                        result_rows.append({
                            "sequence": sequence, "horizon_H": horizon,
                            "target_variant": target, "budget_fraction": budget,
                            "n_candidates": n, "strategy": strategy,
                            "selected_anchor_frames": ",".join(map(str, anchor_arr[idx].tolist())),
                            **sc,
                        })

                    # Add random expectation as a strategy row for simple comparisons.
                    result_rows.append({
                        "sequence": sequence, "horizon_H": horizon,
                        "target_variant": target, "budget_fraction": budget,
                        "n_candidates": n, "strategy": "random_mean_over_trials",
                        "selected_anchor_frames": "",
                        "selected_n": k,
                        "selected_total_gain": float(np.mean(random_totals)),
                        "selected_mean_gain": float(np.mean(random_means)),
                        "selected_positive_fraction": np.nan,
                        "selected_negative_count": np.nan,
                        "oracle_topk_total_gain": float(np.sum(y[oracle_idx])),
                        "regret_vs_oracle_topk": float(np.sum(y[oracle_idx]) - np.mean(random_totals)),
                        "gain_per_selected_anchor": float(np.mean(random_totals) / k),
                        "beats_no_refresh": bool(np.mean(random_totals) > 0),
                        "no_refresh_total_gain": 0.0,
                    })

    results = pd.DataFrame(result_rows)
    random_df = pd.DataFrame(random_rows)
    audit_df = pd.DataFrame(audit_rows)
    if results.empty:
        raise RuntimeError("No ranking results generated. Check target names and input predictions.")

    results.to_csv(out_dir / "p2_2_3_budget_results_per_sequence.csv", index=False, encoding="utf-8-sig")
    random_df.to_csv(out_dir / "p2_2_3_random_baseline_distribution.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(out_dir / "p2_2_3_candidate_audit.csv", index=False, encoding="utf-8-sig")

    # Macro summary: equal weight per sequence; do not let longer sequences dominate.
    strat = results[results["strategy"].str.startswith("rank_") |
                    results["strategy"].isin(["temporal_even_spacing", "random_mean_over_trials", "no_refresh", "oracle_topk"])]
    macro = strat.groupby(["target_variant", "horizon_H", "budget_fraction", "strategy"], as_index=False).agg(
        sequences=("sequence", "nunique"),
        mean_selected_gain=("selected_mean_gain", "mean"),
        mean_total_gain=("selected_total_gain", "mean"),
        mean_regret_vs_oracle=("regret_vs_oracle_topk", "mean"),
        mean_positive_fraction=("selected_positive_fraction", "mean"),
        mean_negative_count=("selected_negative_count", "mean"),
    )
    macro.to_csv(out_dir / "p2_2_3_budget_results_macro.csv", index=False, encoding="utf-8-sig")

    # Rank-based lift relative to random and temporal baselines per sequence/budget.
    comparison_rows = []
    for (target, h, budget, seq), group in results.groupby(
        ["target_variant", "horizon_H", "budget_fraction", "sequence"]
    ):
        by_strategy = group.set_index("strategy")
        random_gain = by_strategy.loc["random_mean_over_trials", "selected_total_gain"] if "random_mean_over_trials" in by_strategy.index else np.nan
        temporal_gain = by_strategy.loc["temporal_even_spacing", "selected_total_gain"] if "temporal_even_spacing" in by_strategy.index else np.nan
        for strategy in [s for s in by_strategy.index if s.startswith("rank_")]:
            r = by_strategy.loc[strategy]
            comparison_rows.append({
                "sequence": seq, "horizon_H": h, "target_variant": target,
                "budget_fraction": budget, "strategy": strategy,
                "selected_total_gain": r["selected_total_gain"],
                "random_mean_total_gain": random_gain,
                "lift_vs_random_total_gain": r["selected_total_gain"] - random_gain,
                "temporal_total_gain": temporal_gain,
                "lift_vs_temporal_total_gain": r["selected_total_gain"] - temporal_gain,
                "regret_vs_oracle_topk": r["regret_vs_oracle_topk"],
                "beats_random_mean": bool(r["selected_total_gain"] > random_gain),
                "beats_temporal": bool(r["selected_total_gain"] > temporal_gain),
            })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(out_dir / "p2_2_3_model_vs_baselines.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# SVR-Depth P2.2.3 Fixed-Budget Ranking Report", "",
        f"- Input: `{input_dir / 'p2_2_2_predictions.csv'}`",
        f"- Random trials per sequence/horizon/budget: {args.random_trials}",
        f"- Budgets: {', '.join(f'{b:.0%}' for b in budgets)}",
        "- Uses held-out-sequence predictions from P2.2.2.",
        "- Oracle is a diagnostic upper bound selecting top-k anchors by true target; it is not deployable.",
        "- Temporal baseline selects approximately evenly spaced anchors.",
        "- Random baseline uses the same number of selected anchors and reports Monte Carlo distribution.",
        "- Target is offline full-frame state-writeback gain under P1.5 forced propagation, not online B0/B1/B2 utility.",
        "",
        "## Candidate counts", "",
    ]
    if not audit_df.empty:
        lines += ["| Sequence | H | Target | Before finite filter | After filter | Dropped |",
                  "|---|---:|---|---:|---:|---:|"]
        for _, r in audit_df.iterrows():
            lines.append(
                f"| {r.sequence} | {int(r.horizon_H)} | {r.target_variant} | "
                f"{int(r.candidate_anchors_before_finite_filter)} | {int(r.anchors_after_finite_filter)} | "
                f"{int(r.dropped_nonfinite_rows)} |"
            )
    lines += [
        "",
        "## Macro summary: primary target",
        "",
        "| H | Budget | Strategy | Sequences | Mean selected gain | Mean total gain | Mean oracle regret |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    primary_macro = macro[macro.target_variant == PRIMARY_TARGET]
    for _, r in primary_macro.iterrows():
        def fmt(x):
            return "NA" if pd.isna(x) else f"{x:.6g}"
        lines.append(
            f"| {int(r.horizon_H)} | {r.budget_fraction:.0%} | {r.strategy} | "
            f"{int(r.sequences)} | {fmt(r.mean_selected_gain)} | {fmt(r.mean_total_gain)} | "
            f"{fmt(r.mean_regret_vs_oracle)} |"
        )
    lines += [
        "",
        "## Model comparison vs random and temporal baselines",
        "",
        "Positive lift means the model-selected anchors yielded higher true total gain than the comparator.",
        "",
        "| H | Budget | Sequence | Strategy | Lift vs random | Lift vs temporal | Beats random? | Beats temporal? |",
        "|---:|---:|---|---|---:|---:|---|---|",
    ]
    for _, r in comparisons[comparisons.target_variant == PRIMARY_TARGET].iterrows():
        lines.append(
            f"| {int(r.horizon_H)} | {r.budget_fraction:.0%} | {r.sequence} | {r.strategy} | "
            f"{r.lift_vs_random_total_gain:.6g} | {r.lift_vs_temporal_total_gain:.6g} | "
            f"{r.beats_random_mean} | {r.beats_temporal} |"
        )
    lines += [
        "",
        "## Interpretation rules", "",
        "1. Primary evidence is per-sequence lift over random and temporal baselines, not pooled MAE.",
        "2. A useful ranking model should beat both baselines at several budgets and on more than one sequence.",
        "3. Oracle regret is descriptive only; it is an unattainable upper bound.",
        "4. Fixed budget forces exactly k refresh selections, even if some selected gains are negative; compare against no-refresh separately.",
        "5. This does not model B1/B2 compute costs or state interactions between refreshes. It is a first offline ranking test only.",
        "6. The normalized target is sensitivity analysis; do not choose a target post hoc solely because it looks better.",
        "",
        "## Outputs", "",
        "- `p2_2_3_budget_results_per_sequence.csv`",
        "- `p2_2_3_budget_results_macro.csv`",
        "- `p2_2_3_random_baseline_distribution.csv`",
        "- `p2_2_3_model_vs_baselines.csv`",
        "- `p2_2_3_candidate_audit.csv`",
        "- `p2_2_3_report.md`",
    ]
    (out_dir / "p2_2_3_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("=== P2.2.3 complete ===")
    print("Per-sequence strategy rows:", len(results))
    print("Random distributions:", len(random_df))
    print("Output directory:", out_dir)
    print("\nPrimary-target model vs baselines (per-sequence rows):")
    primary_comp = comparisons[comparisons.target_variant == PRIMARY_TARGET]
    if not primary_comp.empty:
        cols = ["horizon_H", "budget_fraction", "sequence", "strategy",
                "lift_vs_random_total_gain", "lift_vs_temporal_total_gain",
                "beats_random_mean", "beats_temporal"]
        print(primary_comp[cols].to_string(index=False))


if __name__ == "__main__":
    main()
