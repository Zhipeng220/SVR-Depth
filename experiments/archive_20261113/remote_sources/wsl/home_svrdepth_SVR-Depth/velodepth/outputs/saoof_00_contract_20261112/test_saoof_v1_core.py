import torch
from saoof_v1_core import keyframe_anchor, decompose, fuse

def main():
    torch.manual_seed(1)
    p = torch.exp(torch.randn(64, 64) * .2)
    ref = p * 2
    b = keyframe_anchor(p, ref)
    assert torch.allclose(b, torch.tensor(-2.).abs().log().neg())
    m, local, valid, reason = decompose(p, ref, b, torch.ones_like(p, dtype=torch.bool))
    assert reason is None
    zero, one = torch.zeros_like(p), torch.ones_like(p)
    assert torch.allclose(fuse(p, m, local, valid, 0., zero), p, atol=1e-6)
    assert torch.allclose(fuse(p, m, local, valid, 1., one), ref * b.exp(), atol=1e-6)
    r2 = ref.clone(); r2[:32] *= 1.2
    m2, loc2, valid2, reason = decompose(p, r2, b, torch.ones_like(p, dtype=torch.bool))
    assert reason is None and torch.allclose(fuse(p, m2, loc2, valid2, 1., one), r2 * b.exp(), atol=1e-6)
    assert torch.equal(fuse(p), p)
    assert keyframe_anchor(p, torch.full_like(p, float('nan'))) is None
    assert decompose(p, ref, None, one)[3] == 'invalid_anchor'
    m3,l3,v3,reason = decompose(p, ref, b, zero.bool())
    assert reason == 'insufficient_corr_mask' and torch.equal(fuse(p,m3,l3,v3,0.,zero),p)
    print('PASS: anchor, dual decomposition, gate endpoints and exact fallback')

if __name__ == '__main__': main()
