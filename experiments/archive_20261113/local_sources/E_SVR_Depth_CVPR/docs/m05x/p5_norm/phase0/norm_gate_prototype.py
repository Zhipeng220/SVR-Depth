"""Pure Python decision prototype; no model, labels or persistent state."""
import math
from numbers import Real

def norm_guided_commit(predicted_utility):
    """Return True iff finite real predicted utility is strictly positive."""
    if isinstance(predicted_utility, bool) or not isinstance(predicted_utility, Real):
        raise TypeError('predicted_utility must be a real scalar, excluding bool')
    if not math.isfinite(predicted_utility):
        raise ValueError('predicted_utility must be finite')
    return predicted_utility > 0

def norm_guided_commit_batch(predicted_utilities):
    """Validate and decide every scalar, raising on any invalid item."""
    if not isinstance(predicted_utilities, (list, tuple)):
        raise TypeError('batch must be a list or tuple of real scalars')
    return [norm_guided_commit(value) for value in predicted_utilities]
