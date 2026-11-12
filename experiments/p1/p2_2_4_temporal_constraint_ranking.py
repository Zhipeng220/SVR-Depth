#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVR-Depth P2.2.4: temporally constrained fixed-budget ranking evaluation.

Input: out-of-sequence predictions from P2.2.2.
No GPU, retraining, or VeloDepth inference required.

Example:
python experiments/p1/p2_2_4_temporal_constraint_ranking.py \
  --input-dir outputs/p2_2_2 \
  --out-dir outputs/p2_2_4 \
  --random-trials 1000 \
  --min-gaps 0,4,8,16,32 \
  --budgets 0.25,0.50,0.75

Frame-gap meanings:
- 0: unconstrained selection (P2.2.3-style reference).
- positive values: any two selected anchor_frame IDs must differ by at least that many source frames.
- Since candidate anchors are usually 4 frames apart, gap=4 is a sanity check; 8/16/32 increasingly reduce temporal overlap.
If a requested budget is infeasible under a gap, all strategies are compared at the maximum feasible k and the effective budget is recorded.
"""
import argparse
from bisect import bisect_left, bisect_right
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SEQUENCES = ("crowd2", "balloon2", "crowd3")
HORIZONS = (1, 4, 8, 16)
PRIMARY_TARGET = "G_state_absrel_discounted_per_valid_frame"
TARGET_MAP = {
    "primary": PRIMARY_TARGET,
    "normalized": "target_discount_weight_normalized",
}
MODELS = ("learned_state_only", "motion_only", "motion_plus_learned")
DEFAULT_GAPS = (0, 4, 8, 16, 32)
DEFAULT_BUDGETS = (0.25, 0.50, 0.75)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input file missing: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def max_feasible_k(anchors, min_gap):
    """Maximum cardinality subset with pairwise anchor distance >= min_gap."""
    if len(anchors) == 0:
        return 0
    if min_gap <= 0:
        return len(anchors)
    count, last = 0, None
    for anchor in anchors:
        if last is None or anchor - last >= min_gap:
            count += 1
            last = anchor
    return count


def select_k_dp(anchors, values, k, min_gap, temporal_targets=None):
    """
    Exact cardinality-constrained dynamic programming.
    If temporal_targets is None, maximize sum(values[selected]).
    Otherwise maximize negative squared distance between selected anchors
    and temporal_targets in order, yielding an evenly spread selection.
    """
    anchors = np.asarray(anchors, dtype=int)
    values = np.asarray(values, dtype=float)
    n = len(anchors)
    if k <= 0:
        return np.asarray([], dtype=int)
    if k > n or k > max_feasible_k(anchors, min_gap):
        raise ValueError(f"Infeasible k={k} for n={n}, gap={min_gap}")
    # pred[row] is the number of candidates that may precede candidate row-1.
    pred = np.zeros(n + 1, dtype=int)
    for idx, anchor in enumerate(anchors):
        if min_gap <= 0:
            pred[idx + 1] = idx
        else:
            pred[idx + 1] = bisect_right(anchors, int(anchor) - min_gap, 0, idx)
    neg_inf = -1e100
    dp = np.full((n + 1, k + 1), neg_inf, dtype=float)
    take = np.zeros((n + 1, k + 1), dtype=bool)
    dp[:, 0] = 0.0
    for row in range(1, n + 1):
        idx = row - 1
        dp[row, :] = dp[row - 1, :]
        for j in range(1, min(k, row) + 1):
            if temporal_targets is None:
                reward = values[idx]
            else:
                reward = -float(anchors[idx] - temporal_targets[j - 1]) ** 2
            base = dp[pred[row], j - 1]
            candidate = base + reward
            if candidate > dp[row, j]:
                dp[row, j] = candidate
                take[row, j] = True
    if not np.isfinite(dp[n, k]) or dp[n, k] <= neg_inf / 2:
        raise ValueError(f"Could not construct k={k} selection at gap={min_gap}")
    chosen = []
    row, j = n, k
    while j > 0 and row > 0:
        if take[row, j]:
            idx = row - 1
            chosen.append(idx)
            row = pred[row]
            j -= 1
        else:
            row -= 1
    if j != 0:
        raise RuntimeError("DP backtracking failed")
    return np.asarray(sorted(chosen), dtype=int)


def random_feasible_subset(anchors, k, min_gap, rng):
    """Uniformly sample a feasible k-subset using counts of feasible completions."""
    anchors = tuple(int(x) for x in anchors)
    n = len(anchors)

    @lru_cache(None)
    def ways(start, remaining):
        if remaining == 0:
            return 1
        if start >= n or n - start < remaining:
            return 0
        # Skip current candidate.
        total = ways(start + 1, remaining)
        # Include current candidate and jump to first allowed next candidate.
        if min_gap <= 0:
            nxt = start + 1
        else:
            nxt = bisect_left(anchors, anchors[start] + min_gap, start + 1)
        total += ways(nxt, remaining - 1)
        return total

    if ways(0, k) <= 0:
        raise ValueError(f"No feasible random subset for k={k}, gap={min_gap}")
    chosen = []
    start, remaining = 0, k
    while remaining:
        if start >= n:
            raise RuntimeError("Random subset sampler exhausted candidates")
        skip_count = ways(start + 1, remaining)
        if min_gap <= 0:
            nxt = start + 1
        else:
            nxt = bisect_left(anchors, anchors[start] + min_gap, start + 1)
        take_count = ways(nxt, remaining - 1)
        total = skip_count + take_count
        take_current = take_count > 0 and (skip_count == 0 or rng.random() < take_count / total)
        if take_current:
            chosen.append(start)
            start = nxt
            remaining -= 1
        else:
            start += 1
    return np.asarray(chosen, dtype=int)


def score_selection(y, selected_idx, oracle_total):
    y = np.asarray(y, dtype=float)
    idx = np.asarray(selected_idx, dtype=int)
    vals = y[idx]
    total = float(np.sum(vals))
    return {
        "selected_n": int(len(idx)),
        "selected_total_gain": total,
        "selected_mean_gain": float(np.mean(vals)) if len(vals) else np.nan,
        "selected_positive_fraction": float(np.mean(vals > 0)) if len(vals) else np.nan,
        "selected_negative_count": int(np.sum(vals <= 0)),
        "oracle_constrained_total_gain": float(oracle_total),
        "regret_vs_constrained_oracle": float(oracle_total - total),
        "beats_no_refresh": bool(total > 0.0),
        "no_refresh_total_gain": 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="P2.2.4 temporally constrained fixed-budget ranking")
    ap.add_argument("--input-dir", default="outputs/p2_2_2")
    ap.add_argument("--out-dir", default="outputs/p2_2_4")
    ap.add_argument("--random-trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--min-gaps", default="0,4,8,16,32")
    ap.add_argument("--budgets", default="0.25,0.50,0.75")
    ap.add_argument("--target-variants", default="primary,normalized")
    args = ap.parse_args()

    if args.random_trials < 100:
        ap.error("--random-trials should be at least 100")
    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = input_dir / "p2_2_2_predictions.csv"
    pred = read_csv(pred_path)
    required = {"sequence", "anchor_frame", "horizon_H", "target_variant", "model", "y_true", "y_pred"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"{pred_path} missing columns: {sorted(missing)}")

    gaps = sorted(set(int(x.strip()) for x in args.min_gaps.split(",") if x.strip()))
    budgets = sorted(set(float(x.strip()) for x in args.budgets.split(",") if x.strip()))
    target_keys = [x.strip() for x in args.target_variants.split(",") if x.strip()]
    if not gaps or any(g < 0 for g in gaps):
        ap.error("--min-gaps must be nonnegative integers")
    if not budgets or any(not 0 < b <= 1 for b in budgets):
        ap.error("--budgets values must be in (0,1]")
    try:
        targets = [TARGET_MAP[x] for x in target_keys]
    except KeyError as exc:
        ap.error(f"Unknown target variant {exc}; use primary,normalized")

    pred = pred[pred.model.isin(MODELS)].copy()
    pred["anchor_frame"] = pd.to_numeric(pred.anchor_frame, errors="raise").astype(int)
    pred["horizon_H"] = pd.to_numeric(pred.horizon_H, errors="raise").astype(int)
    pred["y_true"] = pd.to_numeric(pred.y_true, errors="coerce")
    pred["y_pred"] = pd.to_numeric(pred.y_pred, errors="coerce")
    key = ["sequence", "anchor_frame", "horizon_H", "target_variant", "model"]
    if pred.duplicated(key).any():
        raise ValueError("Duplicate (sequence, anchor, horizon, target, model) rows found; refusing ambiguous evaluation.")

    rng = np.random.default_rng(args.seed)
    results, random_rows, audit_rows, comparison_rows = [], [], [], []
    for target in targets:
        for horizon in HORIZONS:
            for seq in SEQUENCES:
                sub = pred[(pred.target_variant == target) &
                           (pred.horizon_H == horizon) &
                           (pred.sequence == seq)].copy()
                if sub.empty:
                    audit_rows.append({"sequence": seq, "horizon_H": horizon, "target_variant": target,
                                       "status": "missing_group", "candidate_count": 0})
                    continue
                pp = sub.pivot(index="anchor_frame", columns="model", values="y_pred")
                yy = sub.groupby("anchor_frame")["y_true"].first()
                missing_models = [m for m in MODELS if m not in pp.columns]
                if missing_models:
                    raise ValueError(f"{seq} H={horizon} missing models: {missing_models}")
                anchors_all = sorted(set(pp.index) & set(yy.index))
                pp = pp.loc[anchors_all, list(MODELS)]
                y_all = yy.loc[anchors_all].to_numpy(float)
                pred_all = pp.to_numpy(float)
                finite = np.isfinite(y_all) & np.all(np.isfinite(pred_all), axis=1)
                anchors = np.asarray(anchors_all, dtype=int)[finite]
                y = y_all[finite]
                pred_matrix = pred_all[finite]
                if len(anchors) < 3:
                    audit_rows.append({"sequence": seq, "horizon_H": horizon, "target_variant": target,
                                       "status": "too_few_candidates", "candidate_count": len(anchors)})
                    continue
                min_spacing = int(np.min(np.diff(anchors))) if len(anchors) > 1 else 0
                median_spacing = float(np.median(np.diff(anchors))) if len(anchors) > 1 else 0.0

                for gap in gaps:
                    maxk = max_feasible_k(anchors, gap)
                    for budget in budgets:
                        requested_k = min(len(anchors), max(1, int(np.ceil(budget * len(anchors)))))
                        k = min(requested_k, maxk)
                        if k <= 0:
                            continue
                        effective_budget = k / len(anchors)
                        budget_feasible = requested_k <= maxk
                        oracle_idx = select_k_dp(anchors, y, k, gap)
                        oracle_total = float(np.sum(y[oracle_idx]))
                        temporal_targets = np.linspace(float(anchors[0]), float(anchors[-1]), num=k)
                        temporal_idx = select_k_dp(
                            anchors, np.zeros(len(anchors)), k, gap, temporal_targets=temporal_targets
                        )
                        random_totals, random_means = [], []
                        for _ in range(args.random_trials):
                            ridx = random_feasible_subset(anchors, k, gap, rng)
                            random_totals.append(float(np.sum(y[ridx])))
                            random_means.append(float(np.mean(y[ridx])))
                        random_totals = np.asarray(random_totals, dtype=float)
                        random_means = np.asarray(random_means, dtype=float)
                        random_mean_total = float(np.mean(random_totals))
                        random_summary = {
                            "sequence": seq, "horizon_H": horizon, "target_variant": target,
                            "min_gap_frames": gap, "requested_budget_fraction": budget,
                            "effective_budget_fraction": effective_budget, "n_candidates": len(anchors),
                            "requested_k": requested_k, "selected_k": k, "budget_feasible": budget_feasible,
                            "random_trials": args.random_trials,
                            "random_total_gain_mean": random_mean_total,
                            "random_total_gain_std": float(np.std(random_totals, ddof=1)),
                            "random_total_gain_p05": float(np.quantile(random_totals, 0.05)),
                            "random_total_gain_p50": float(np.quantile(random_totals, 0.50)),
                            "random_total_gain_p95": float(np.quantile(random_totals, 0.95)),
                            "random_mean_gain_mean": float(np.mean(random_means)),
                            "random_probability_beats_no_refresh": float(np.mean(random_totals > 0)),
                        }
                        random_rows.append(random_summary)
                        strategy_selections = {
                            "temporal_even_spacing": temporal_idx,
                            "oracle_constrained": oracle_idx,
                        }
                        for j, model in enumerate(MODELS):
                            strategy_selections[f"rank_{model}"] = select_k_dp(
                                anchors, pred_matrix[:, j], k, gap
                            )
                        # Same effective k for all strategies; random is represented by its expectation.
                        for strategy, idx in strategy_selections.items():
                            sc = score_selection(y, idx, oracle_total)
                            row = {
                                "sequence": seq, "horizon_H": horizon, "target_variant": target,
                                "min_gap_frames": gap, "requested_budget_fraction": budget,
                                "effective_budget_fraction": effective_budget, "n_candidates": len(anchors),
                                "requested_k": requested_k, "selected_k": k,
                                "budget_feasible": budget_feasible, "strategy": strategy,
                                "selected_anchor_frames": ",".join(str(int(anchors[i])) for i in idx),
                                **sc,
                            }
                            results.append(row)
                        results.append({
                            "sequence": seq, "horizon_H": horizon, "target_variant": target,
                            "min_gap_frames": gap, "requested_budget_fraction": budget,
                            "effective_budget_fraction": effective_budget, "n_candidates": len(anchors),
                            "requested_k": requested_k, "selected_k": k,
                            "budget_feasible": budget_feasible, "strategy": "no_refresh",
                            "selected_anchor_frames": "", "selected_n": 0,
                            "selected_total_gain": 0.0, "selected_mean_gain": np.nan,
                            "selected_positive_fraction": np.nan, "selected_negative_count": 0,
                            "oracle_constrained_total_gain": oracle_total,
                            "regret_vs_constrained_oracle": oracle_total,
                            "beats_no_refresh": False, "no_refresh_total_gain": 0.0,
                        })
                        results.append({
                            "sequence": seq, "horizon_H": horizon, "target_variant": target,
                            "min_gap_frames": gap, "requested_budget_fraction": budget,
                            "effective_budget_fraction": effective_budget, "n_candidates": len(anchors),
                            "requested_k": requested_k, "selected_k": k,
                            "budget_feasible": budget_feasible, "strategy": "random_mean_over_trials",
                            "selected_anchor_frames": "", "selected_n": k,
                            "selected_total_gain": random_mean_total,
                            "selected_mean_gain": float(np.mean(random_means)),
                            "selected_positive_fraction": np.nan, "selected_negative_count": np.nan,
                            "oracle_constrained_total_gain": oracle_total,
                            "regret_vs_constrained_oracle": oracle_total - random_mean_total,
                            "beats_no_refresh": bool(random_mean_total > 0),
                            "no_refresh_total_gain": 0.0,
                        })
                        # Record model lift over random/temporal with matched k.
                        by_name = {r["strategy"]: r for r in results[-(len(strategy_selections)+2):]}
                        for model in MODELS:
                            name = f"rank_{model}"
                            mr = by_name[name]
                            tr = by_name["temporal_even_spacing"]
                            comparison_rows.append({
                                "sequence": seq, "horizon_H": horizon, "target_variant": target,
                                "min_gap_frames": gap, "requested_budget_fraction": budget,
                                "effective_budget_fraction": effective_budget, "n_candidates": len(anchors),
                                "requested_k": requested_k, "selected_k": k, "budget_feasible": budget_feasible,
                                "model_strategy": name,
                                "model_total_gain": mr["selected_total_gain"],
                                "random_mean_total_gain": random_mean_total,
                                "lift_vs_random": mr["selected_total_gain"] - random_mean_total,
                                "temporal_total_gain": tr["selected_total_gain"],
                                "lift_vs_temporal": mr["selected_total_gain"] - tr["selected_total_gain"],
                                "oracle_constrained_total_gain": oracle_total,
                                "regret_vs_constrained_oracle": mr["regret_vs_constrained_oracle"],
                                "beats_random_mean": bool(mr["selected_total_gain"] > random_mean_total),
                                "beats_temporal": bool(mr["selected_total_gain"] > tr["selected_total_gain"]),
                            })
                        audit_rows.append({
                            "sequence": seq, "horizon_H": horizon, "target_variant": target,
                            "status": "ok", "candidate_count": len(anchors),
                            "dropped_nonfinite_count": int(len(anchors_all) - len(anchors)),
                            "min_anchor_spacing_frames": min_spacing,
                            "median_anchor_spacing_frames": median_spacing,
                            "min_gap_frames": gap, "max_feasible_k": maxk,
                            "requested_budget_fraction": budget, "requested_k": requested_k,
                            "selected_k": k, "effective_budget_fraction": effective_budget,
                            "budget_feasible": budget_feasible,
                        })

    result_df = pd.DataFrame(results)
    random_df = pd.DataFrame(random_rows)
    audit_df = pd.DataFrame(audit_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    if result_df.empty:
        raise RuntimeError("No results generated; check P2.2.2 prediction file and target names.")

    result_df.to_csv(out_dir / "p2_2_4_results_per_sequence.csv", index=False, encoding="utf-8-sig")
    random_df.to_csv(out_dir / "p2_2_4_random_baseline_distribution.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(out_dir / "p2_2_4_candidate_audit.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(out_dir / "p2_2_4_model_vs_baselines.csv", index=False, encoding="utf-8-sig")

    # Macro summaries: equal weight per sequence; report both requested and effective budgets.
    strat = result_df[result_df.strategy.isin(
        [f"rank_{m}" for m in MODELS] +
        ["temporal_even_spacing", "random_mean_over_trials", "oracle_constrained"]
    )]
    macro = strat.groupby(
        ["target_variant", "horizon_H", "min_gap_frames", "requested_budget_fraction",
         "strategy"], as_index=False
    ).agg(
        sequences=("sequence", "nunique"),
        feasible_sequence_count=("budget_feasible", "sum"),
        mean_effective_budget_fraction=("effective_budget_fraction", "mean"),
        mean_selected_gain=("selected_mean_gain", "mean"),
        mean_total_gain=("selected_total_gain", "mean"),
        mean_regret_vs_oracle=("regret_vs_constrained_oracle", "mean"),
        mean_positive_fraction=("selected_positive_fraction", "mean"),
    )
    macro.to_csv(out_dir / "p2_2_4_results_macro.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# SVR-Depth P2.2.4 Temporal-Constraint Ranking Report", "",
        f"- Input predictions: `{pred_path}`",
        f"- Random trials per group: {args.random_trials}",
        f"- Min-gap frame values: {', '.join(map(str, gaps))}",
        f"- Requested budgets: {', '.join(f'{b:.0%}' for b in budgets)}",
        "- Validation predictions are from P2.2.2 leave-one-sequence-out folds.",
        "- All strategies use the same candidate anchors and same effective selection count k within each comparison.",
        "- When requested budget is infeasible under a gap, k is reduced to the maximum feasible count; requested and effective budgets are both recorded.",
        "- Oracle is a non-deployable constrained upper bound based on true future gains.",
        "- Random baseline samples uniformly among feasible subsets of size k.",
        "- Temporal baseline uses dynamic programming to select k feasible anchors as evenly spaced in source-frame time as possible.",
        "- This remains offline full-frame writeback-gain ranking, not online B0/B1/B2 execution or local spatial refresh.",
        "",
        "## Candidate audit", "",
        f"- Audit rows: {len(audit_df)}",
        f"- Infeasible requested-budget cases: {int((~audit_df['budget_feasible'].fillna(False).astype(bool)).sum()) if 'budget_feasible' in audit_df.columns else 0}",
        "",
        "## Primary-target macro summary", "",
        "| H | Gap (frames) | Requested budget | Effective budget | Strategy | Sequences | Mean selected gain | Mean total gain | Mean oracle regret |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    primary_macro = macro[macro.target_variant == PRIMARY_TARGET]
    def fmt(x):
        return "NA" if pd.isna(x) else f"{x:.6g}"
    for _, r in primary_macro.iterrows():
        lines.append(
            f"| {int(r.horizon_H)} | {int(r.min_gap_frames)} | {r.requested_budget_fraction:.0%} | "
            f"{r.mean_effective_budget_fraction:.1%} | {r.strategy} | {int(r.sequences)} | "
            f"{fmt(r.mean_selected_gain)} | {fmt(r.mean_total_gain)} | {fmt(r.mean_regret_vs_oracle)} |"
        )
    lines += [
        "",
        "## Model lift versus baselines (primary target)", "",
        "| H | Gap | Requested budget | Sequence | Model | Lift vs random | Lift vs temporal | Beats random? | Beats temporal? |",
        "|---:|---:|---:|---|---|---:|---:|---|---|",
    ]
    for _, r in comparison_df[comparison_df.target_variant == PRIMARY_TARGET].iterrows():
        lines.append(
            f"| {int(r.horizon_H)} | {int(r.min_gap_frames)} | {r.requested_budget_fraction:.0%} | "
            f"{r.sequence} | {r.model_strategy} | {fmt(r.lift_vs_random)} | {fmt(r.lift_vs_temporal)} | "
            f"{r.beats_random_mean} | {r.beats_temporal} |"
        )
    lines += [
        "",
        "## Interpretation rules", "",
        "1. Primary evidence is per-sequence lift over constrained random and temporal baselines, not pooled regression error.",
        "2. A useful signal should remain positive at meaningful gaps (especially 8, 16 and 32 frames) and on more than one sequence.",
        "3. Compare only matched effective budgets; infeasible requested budgets are explicitly reduced and flagged.",
        "4. A positive model lift does not prove online utility: future windows can still overlap and the state transition/action costs are not simulated.",
        "5. Treat results as exploratory because there are only three sequences and temporally correlated anchors.",
        "",
        "## Output files", "",
        "- `p2_2_4_results_per_sequence.csv`",
        "- `p2_2_4_results_macro.csv`",
        "- `p2_2_4_random_baseline_distribution.csv`",
        "- `p2_2_4_model_vs_baselines.csv`",
        "- `p2_2_4_candidate_audit.csv`",
        "- `p2_2_4_report.md`",
    ]
    (out_dir / "p2_2_4_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("=== P2.2.4 complete ===")
    print("Per-sequence strategy rows:", len(result_df))
    print("Model-vs-baseline comparisons:", len(comparison_df))
    print("Audit rows:", len(audit_df))
    print("Output directory:", out_dir)
    print("\nPrimary-target comparisons (first 60 rows):")
    if not comparison_df.empty:
        cols = ["horizon_H", "min_gap_frames", "requested_budget_fraction", "effective_budget_fraction",
                "sequence", "model_strategy", "lift_vs_random", "lift_vs_temporal",
                "beats_random_mean", "beats_temporal"]
        print(comparison_df[comparison_df.target_variant == PRIMARY_TARGET][cols].head(60).to_string(index=False))


if __name__ == "__main__":
    main()
