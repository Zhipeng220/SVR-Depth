from pathlib import Path
import hashlib
import torch
from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead
repo=Path.cwd()
ckpt_path=repo/'outputs/gt4a2_y05r_20261029_222944_4625/validation/training/C_star/final.pt'
assert ckpt_path.is_file(), f'Missing checkpoint: {ckpt_path}'
ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=True)
assert ckpt.get('experiment')=='Y05-R', ckpt.get('experiment')
assert ckpt.get('kind')=='C_star', ckpt.get('kind')
assert isinstance(ckpt.get('state'),dict), 'Checkpoint state missing'
state=ckpt['state']
print('checkpoint:',ckpt_path)
print('checkpoint_sha256:',hashlib.sha256(ckpt_path.read_bytes()).hexdigest())
print('experiment/kind/step:',ckpt.get('experiment'),ckpt.get('kind'),ckpt.get('step'))
print('state entries:',len(state))
head=PersistentFeatureRepairHead(channels=512,bottleneck=32,levels=4)
expected=head.state_dict()
if set(state)!=set(expected):
    print('missing keys:',sorted(set(expected)-set(state)))
    print('unexpected keys:',sorted(set(state)-set(expected)))
    raise SystemExit('FAIL: checkpoint keys do not match repair-head implementation')
shape_errors=[(k,tuple(state[k].shape),tuple(expected[k].shape)) for k in expected if tuple(state[k].shape)!=tuple(expected[k].shape)]
if shape_errors:
    print('shape errors:',shape_errors)
    raise SystemExit('FAIL: checkpoint tensor shapes do not match')
head.load_state_dict(state,strict=True)
head.eval()
with torch.no_grad():
    xs=[torch.randn(1,512,h,w) for h,w in [(16,16),(8,8),(4,4),(2,2)]]
    ys=head(xs)
assert [tuple(y.shape) for y in ys]==[tuple(x.shape) for x in xs]
assert all(torch.isfinite(y).all().item() for y in ys)
print('STRICT LOAD: PASS')
print('CPU FORWARD / SHAPES / FINITE: PASS')
print('parameter_count:',sum(p.numel() for p in head.parameters()))
