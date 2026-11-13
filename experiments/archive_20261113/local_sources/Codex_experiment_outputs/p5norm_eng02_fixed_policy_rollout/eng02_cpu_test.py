import torch
from eng02_policy import build_policy_state
a=[torch.ones(2,3)*i for i in range(4)];b=[x+5 for x in a];s={'features':b,'radius':torch.ones(2),'confidence':torch.ones(2),'rays_map':torch.ones(2)}
for policy,src in [('HOLD',a),('FULL_COMMIT',b),('LAYER1_ONLY',[b[0],*a[1:]])]:
 x=build_policy_state(s,a,b,policy);assert all(torch.equal(q,w) and q.untyped_storage().data_ptr()!=w.untyped_storage().data_ptr() for q,w in zip(x['features'],src));x['features'][0].add_(50);assert torch.equal(a[0],torch.zeros_like(a[0])) and torch.equal(b[0],torch.full_like(b[0],5))
for bad in ['BAD']:
 try:build_policy_state(s,a,b,bad)
 except ValueError:pass
 else:raise AssertionError('unsupported policy accepted')
bad=dict(s,other=torch.ones(1))
try:build_policy_state(bad,a,b,'HOLD')
except ValueError:pass
else:raise AssertionError('unknown state accepted')
print('CPU PASS')
