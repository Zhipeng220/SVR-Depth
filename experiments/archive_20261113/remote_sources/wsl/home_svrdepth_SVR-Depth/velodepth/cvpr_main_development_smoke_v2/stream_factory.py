"""Construct only the three Y09 stream families used by the development smoke."""

def build_stream(stream_class, rt, model, shape, method, k, adapter):
    if method not in ('A', 'O_multi', 'P') or k not in (16, 32, 64):
        raise ValueError('Unknown smoke method or schedule')
    if method == 'O_multi' and adapter is None:
        raise ValueError('O_multi requires its loaded output adapter')
    if method in ('A', 'P') and adapter is not None:
        raise ValueError(f'{method} must not receive an output adapter')
    cfg = dict(family=method, schedule='K'+str(k), gain=1., frequency=1)
    return stream_class(rt, model, cfg, shape, adapter=adapter)

def verify_counts(counts, method, k, frames):
    keyframes = (frames + k - 1) // k
    fast = frames - keyframes
    expected = {
        'keyframe_calls': keyframes, 'flow_calls': fast, 'native_fast_calls': fast,
        'repair_head_calls': fast if method == 'P' else 0,
        'output_adapter_calls': fast if method == 'O_multi' else 0,
        'additional_readout_passes': fast if method == 'O_multi' else 0,
        'rejected_candidates': 0,
    }
    if counts != expected:
        raise RuntimeError(f'Unexpected actual call counts for {method} K{k}: {counts}; expected {expected}')
