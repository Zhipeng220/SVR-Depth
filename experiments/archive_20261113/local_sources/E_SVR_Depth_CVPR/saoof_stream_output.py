"""Per-frame output adapter. Never accepts or returns native propagation state."""
import torch
from saoof_v1_core import keyframe_anchor, decompose, fuse
from saoof_correspondence import correspondence_mask

class SAOOFStreamOutput:
    def __init__(self):
        self._anchor=None

    def begin_cycle(self, native_keyframe, reference_keyframe, geometry_mask=None):
        self._anchor = None if reference_keyframe is None else keyframe_anchor(native_keyframe,reference_keyframe,geometry_mask)
        return self._anchor is not None

    def process(self, native_depth, reference_depth, forward, backward, geometry_mask, gamma, eta):
        if reference_depth is None:
            return native_depth.clone(), 'missing_reference'
        if self._anchor is None:
            return native_depth.clone(), 'invalid_anchor'
        if forward is None or backward is None:
            return native_depth.clone(), 'missing_flow'
        corr,_=correspondence_mask(forward,backward)
        corr=torch.from_numpy(corr).to(device=native_depth.device)
        result=decompose(native_depth,reference_depth,self._anchor,corr,geometry_mask)
        output=fuse(native_depth,result['global_residual'],result['local_residual'],result['valid_depth_mask'],gamma,eta)
        return output,result['status']
