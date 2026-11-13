"""M04 continuous branch semantics over the repository's Y10 Engine API."""
ARMS=('OFF','OUTPUT_ONLY','F_ONLY','P_NATIVE')

def branch_step(arm,index,k,step,clone):
    """step(repeats) must evaluate from the same unchanged parent state.

    The two diagnostics use two full fast steps, not merely two readouts.
    Returns the selected state, CPU depth and physical fast-step count.
    """
    if arm not in ARMS: raise ValueError('Unknown arm')
    if index % k == 0:
        state,depth=step(0)
        return clone(state),depth,0
    if arm=='OFF':
        s,d=step(0); return clone(s),d,1
    if arm=='P_NATIVE':
        s,d=step(1); return clone(s),d,1
    native,d0=step(0)
    repaired,d1=step(1)
    if arm=='OUTPUT_ONLY': return clone(native),d1,2
    r,c,_,rays=native
    return clone((r,c,repaired[2],rays)),d0,2
