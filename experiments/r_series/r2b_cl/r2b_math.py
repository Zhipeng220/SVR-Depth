"""Pure CPU math helpers for R2-B-CL."""
import math

GAMMA = 0.9

def discounted_gain(discard_by_k, commit_by_k, horizon, gamma=GAMMA):
    """Return discounted AbsRel gain sum and number of valid paired GT frames.

    G_H = sum_{k=1..H} gamma**k * (L_discard[t+k] - L_commit[t+k]).
    Missing GT is represented by NaN on both branches. The original exponent k
    is preserved; invalid frames are not re-indexed.
    """
    total, valid = 0.0, 0
    for k in range(1, horizon + 1):
        d = float(discard_by_k.get(k, float("nan")))
        c = float(commit_by_k.get(k, float("nan")))
        fd, fc = math.isfinite(d), math.isfinite(c)
        if fd != fc:
            raise RuntimeError(f"Counterfactual metric masks differ at future offset k={k}: discard={d}, commit={c}")
        if fd:
            total += (gamma ** k) * (d - c)
            valid += 1
    if valid == 0:
        raise RuntimeError("No valid paired future GT frames in oracle horizon")
    return total, valid

