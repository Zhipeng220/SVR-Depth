#!/usr/bin/env python3
"""Recompute R2-B offline horizon diagnostics from the three R2-A review ZIPs."""
from pathlib import Path
import zipfile, json, argparse
import pandas as pd
import numpy as np

PLANS = ["000","001","010","011","100","101","110","111"]
ANCHORS = [9,37,65]
HORIZONS = [1,4,8,16]
GAMMA = 0.9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("."))
    ap.add_argument("--output-dir", type=Path, default=Path("r2b_offline_horizon_analysis"))
    args = ap.parse_args()
    sources = {
        "crowd2": args.input_dir / "crowd2_86_20261021T113059_406855Z_review.zip",
        "crowd3": args.input_dir / "crowd3_86_20261021T113222_488077Z_review.zip",
        "balloon2": args.input_dir / "balloon2_86_20261021T113140_069566Z_review.zip",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    h16_checks = {}
    for seq, path in sources.items():
        with zipfile.ZipFile(path) as z:
            traj = pd.read_csv(z.open("results/trajectory_summary.csv"), dtype={"plan":str}).set_index("plan")
            original = pd.read_csv(z.open("results/paired_commit_effects.csv"),
                                   dtype={"discard_plan":str,"commit_plan":str})
            frames = {p: pd.read_csv(z.open(f"results/frame_metrics_{p}.csv"),
                                     dtype={"plan":str}).set_index("frame") for p in PLANS}
            for anchor, j in zip(ANCHORS, range(3)):
                for p0 in PLANS:
                    if p0[j] != "0": continue
                    p1 = p0[:j] + "1" + p0[j+1:]
                    a, b = frames[p0], frames[p1]
                    for t in range(anchor + 1):
                        x, y = float(a.loc[t,"absrel"]), float(b.loc[t,"absrel"])
                        if np.isfinite(x) and np.isfinite(y) and abs(x-y) > 1e-6:
                            raise RuntimeError(f"paired prefix mismatch: {seq}/{anchor}/{p0}/{p1}/{t}")
                    full_gain = float(traj.loc[p0,"mean_absrel"] - traj.loc[p1,"mean_absrel"])
                    for H in HORIZONS:
                        terms = []
                        for k in range(1,H+1):
                            t = anchor+k
                            x, y = float(a.loc[t,"absrel"]), float(b.loc[t,"absrel"])
                            if np.isfinite(x) and np.isfinite(y):
                                terms.append((k,x-y))
                        if len(terms) != H:
                            raise RuntimeError(f"unexpected missing GT in selected horizon: {seq}/{anchor}/{H}")
                        gain = sum(GAMMA**k*g for k,g in terms)
                        predicted_commit, hindsight_commit = gain > 0, full_gain > 0
                        match = predicted_commit == hindsight_commit
                        rows.append({
                            "sequence":seq, "anchor_frame_zero_based":anchor,
                            "discard_plan":p0, "commit_plan":p1, "H":H, "gamma":GAMMA,
                            "valid_future_pairs":len(terms),
                            "discounted_future_absrel_gain_sum":gain,
                            "discounted_future_absrel_gain_per_valid_frame":gain/len(terms),
                            "horizon_action":"COMMIT" if predicted_commit else "DISCARD",
                            "full_clip_preferred_action":"COMMIT" if hindsight_commit else "DISCARD",
                            "horizon_matches_full_clip_sign":match,
                            "full_clip_absrel_gain_discard_minus_commit":full_gain,
                            "full_clip_diagnostic_regret":0.0 if match else abs(full_gain),
                        })
        d = pd.DataFrame([r for r in rows if r["sequence"] == seq and r["H"] == 16])
        d = d.sort_values(["anchor_frame_zero_based","discard_plan"])
        source = original.sort_values(["anchor","discard_plan"])
        h16_checks[seq] = float(np.max(np.abs(
            d["discounted_future_absrel_gain_sum"].to_numpy() -
            source["discounted_future_absrel_gain_sum"].to_numpy()
        )))
        if h16_checks[seq] > 1e-10:
            raise RuntimeError(f"H=16 reproduction failed for {seq}: {h16_checks[seq]}")
    pair = pd.DataFrame(rows)
    pair.to_csv(args.output_dir/"r2b_horizon_pairwise.csv",index=False,encoding="utf-8-sig")
    summary = pair.groupby(["sequence","anchor_frame_zero_based","H"]).agg(
        context_pairs=("horizon_action","size"),
        mean_discounted_gain_sum=("discounted_future_absrel_gain_sum","mean"),
        mean_discounted_gain_per_valid_frame=("discounted_future_absrel_gain_per_valid_frame","mean"),
        positive_context_pairs=("discounted_future_absrel_gain_sum",lambda s:int((s>0).sum())),
        negative_context_pairs=("discounted_future_absrel_gain_sum",lambda s:int((s<0).sum())),
        matches_full_clip_action=("horizon_matches_full_clip_sign","sum"),
        full_clip_action_mismatches=("horizon_matches_full_clip_sign",lambda s:int((~s).sum())),
    ).reset_index()
    summary.to_csv(args.output_dir/"r2b_horizon_anchor_summary.csv",index=False,encoding="utf-8-sig")
    transitions=[]
    for h0,h1 in [(1,4),(4,8),(8,16),(1,16)]:
        a=pair[pair.H==h0].set_index(["sequence","anchor_frame_zero_based","discard_plan"])["horizon_action"]
        b=pair[pair.H==h1].set_index(["sequence","anchor_frame_zero_based","discard_plan"])["horizon_action"]
        diff=a!=b
        transitions.append({"H_from":h0,"H_to":h1,"paired_contexts":len(diff),
                            "action_changes":int(diff.sum()),"action_change_fraction":float(diff.mean())})
    pd.DataFrame(transitions).to_csv(args.output_dir/"r2b_horizon_action_transitions.csv",
                                     index=False,encoding="utf-8-sig")
    decisions=pair.groupby("H").agg(
        paired_contexts=("horizon_action","size"),
        full_clip_sign_matches=("horizon_matches_full_clip_sign","sum"),
        full_clip_sign_match_rate=("horizon_matches_full_clip_sign","mean"),
        total_in_sample_diagnostic_regret=("full_clip_diagnostic_regret","sum")
    ).reset_index()
    decisions.to_csv(args.output_dir/"r2b_horizon_decision_summary.csv",index=False,encoding="utf-8-sig")
    print("H=16 reproduction max abs diffs:",json.dumps(h16_checks))
    print("Horizon transition summary:")
    print(pd.read_csv(args.output_dir/"r2b_horizon_action_transitions.csv").to_string(index=False))
if __name__ == "__main__":
    main()
