"""Pure state selection for same-head continuous closed-loop controls."""
from __future__ import annotations
from svr_depth.streaming.state import StreamState

CLOSED_LOOP_BRANCHES = ('A_off', 'O_same_head', 'F_only', 'P_full')

def select_closed_loop(branch: str, native: StreamState, repaired: StreamState) -> tuple[StreamState, StreamState]:
    """Return independent `(visible, committed)` state selections.

    The caller is responsible for evaluating native and repaired candidates
    from the same parent and for handling model RNG and additional call costs.
    """
    if branch not in CLOSED_LOOP_BRANCHES:
        raise ValueError(f'Unknown closed-loop branch: {branch}')
    if branch == 'A_off':
        return native.clone(), native.clone()
    if branch == 'O_same_head':
        return repaired.clone(), native.clone()
    if branch == 'P_full':
        return repaired.clone(), repaired.clone()
    committed = native.clone()
    committed.features = repaired.clone().features
    return native.clone(), committed
