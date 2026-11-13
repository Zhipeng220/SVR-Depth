"""Feature-only writeback. Layer order is the P5-Norm 2.4 repair-hook order."""
import torch

def layer1_writeback(native_state, s0, s1):
    if set(native_state) != {'features', 'radius', 'confidence', 'rays_map'}:
        raise ValueError('unexpected M05 state fields')
    if len(s0) != 4 or len(s1) != 4 or len(native_state['features']) != 4:
        raise ValueError('expected four features')
    for a,b in zip(s0,s1):
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            raise ValueError('repair feature metadata mismatch')
    result={k:v.detach().clone() for k,v in native_state.items() if k!='features'}
    result['features']=[(s1[i] if i==0 else s0[i]).detach().clone() for i in range(4)]
    return result
