import torch
from eng04_policy import build_policy_state
s0=[torch.full((2,3),float(i)) for i in range(4)];s1=[v+5 for v in s0];state={'features':s1,'radius':torch.ones(2),'confidence':torch.ones(2),'rays_map':torch.ones(2)}
for policy in ['FULL_COMMIT','NO_L1','NO_L2','NO_L3','NO_L4']:
 a=build_policy_state(state,s0,s1,policy);removed=int(policy[-1])-1 if policy.startswith('NO_L') else -1
 for i,v in enumerate(a['features']):assert torch.equal(v,s0[i] if i==removed else s1[i])
 a['features'][0].add_(100);assert s0[0][0,0]==0 and s1[0][0,0]==5
print('five policy CPU tests PASS')
