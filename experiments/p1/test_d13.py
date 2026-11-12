import argparse, torch
from d13_rtdp_safe import D13RTDPSafe, harm_gate_loss
def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cpu"); a=p.parse_args()
    dev=torch.device(a.device)
    if dev.type=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    torch.manual_seed(20261030)
    m=D13RTDPSafe(16,8).to(dev)
    x=torch.randn(2,16,32,40,device=dev); z=torch.randn(2,8,32,40,device=dev)
    y=m(x,z)
    assert y["read_feature"].shape==x.shape and y["committed_state"].shape==x.shape
    assert y["alpha"].shape==(2,1,32,40)
    # alpha=0 must leave persistent state exactly unchanged; readout remains separately addressable.
    y0=m(x,z,force_alpha=0.0)
    assert torch.equal(y0["committed_state"],x)
    assert torch.equal(y0["read_feature"],x+y0["delta_read"])
    # alpha=1 must not leak read residual into committed state.
    y1=m(x,z,force_alpha=1.0)
    assert torch.equal(y1["committed_state"],x+y1["delta_trans"])
    loss=harm_gate_loss(y["alpha"],torch.randn(2,1,32,40,device=dev)) + y["read_feature"].square().mean() + y["committed_state"].square().mean() + y["delta_trans"].square().mean()
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    print("D13 module contract PASS; device=",dev,"params=",sum(p.numel() for p in m.parameters()))
if __name__=="__main__": main()
