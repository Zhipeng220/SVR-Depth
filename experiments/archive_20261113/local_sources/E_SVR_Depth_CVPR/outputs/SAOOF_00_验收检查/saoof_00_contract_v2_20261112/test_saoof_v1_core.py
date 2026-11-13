import torch
from saoof_v1_core import keyframe_anchor, decompose, fuse

def main():
    torch.manual_seed(1)
    p = torch.exp(torch.randn(64, 64) * .2)
    ref = p * 2
    b = keyframe_anchor(p, ref)
    assert torch.allclose(b, torch.tensor(-2.).abs().log().neg())
    result = decompose(p, ref, b, torch.ones_like(p, dtype=torch.bool))
    m, local, valid = result['global_residual'], result['local_residual'], result['valid_depth_mask']
    assert result['status'] == 'ok'
    zero, one = torch.zeros_like(p), torch.ones_like(p)
    assert torch.allclose(fuse(p, m, local, valid, 0., zero), p, atol=1e-6)
    assert torch.allclose(fuse(p, m, local, valid, 1., one), ref * b.exp(), atol=1e-6)
    r2 = ref.clone(); r2[:32] *= 1.2
    res2 = decompose(p, r2, b, torch.ones_like(p, dtype=torch.bool))
    assert res2['status'] == 'ok' and torch.allclose(fuse(p, res2['global_residual'], res2['local_residual'], res2['valid_depth_mask'], 1., one), r2 * b.exp(), atol=1e-6)
    assert torch.equal(fuse(p), p)
    assert keyframe_anchor(p, torch.full_like(p, float('nan'))) is None
    assert decompose(p, ref, None, one)['status'] == 'invalid_anchor'
    r3 = decompose(p, ref, b, zero.bool())
    assert r3['status'] == 'insufficient_corr_mask' and not r3['correspondence_mask'].any() and r3['valid_depth_mask'].all()
    assert torch.equal(fuse(p,r3['global_residual'],r3['local_residual'],r3['valid_depth_mask'],0.,zero),p)
    mask = torch.zeros_like(p, dtype=torch.bool); mask[:31] = True
    assert keyframe_anchor(p, ref, mask) is None
    mask[31] = True
    assert keyframe_anchor(p, ref, mask) is not None
    partial = decompose(p, ref, b, one, mask)
    assert torch.equal(partial['valid_depth_mask'], mask) and torch.equal(partial['correspondence_mask'], mask)
    print('PASS: anchor, dual decomposition, gate endpoints and exact fallback')

if __name__ == '__main__': main()
