"""Pure state selections for same-parent intervention experiments."""
from __future__ import annotations
from svr_depth.streaming.state import StreamState

def select_interventions(native: StreamState, repaired: StreamState) -> dict[str, tuple[StreamState, StreamState]]:
    """Return `(visible, committed)` pairs for A, O, F, T and P_native.

    This only selects fields; candidate computation, RNG restoration, anchor
    selection and future replay must be handled separately by an experiment.
    """
    f_only = native.clone()
    f_only.features = repaired.clone().features
    return {
        'A': (native.clone(), native.clone()),
        'O': (repaired.clone(), native.clone()),
        'F': (native.clone(), f_only.clone()),
        'T': (repaired.clone(), f_only.clone()),
        'P_native': (repaired.clone(), repaired.clone()),
    }
