import torch
from eng05_policy import build_policy_state
a=[torch.full((2,3),float(i)) for i in range(4)];b=[x+4 for x in a];st={'features':b,'radius':torch.ones(2),'confidence':torch.ones(2),'rays_map':torch.ones(2)}
for num in range(16):
 mask=format(num,'04b');o=build_policy_state(st,a,b,mask)
 for i,x in enumerate(o['features']):assert torch.equal(x,b[i] if mask[i]=='1' else a[i])
 o['features'][0].add_(30);assert a[0][0,0]==0 and b[0][0,0]==4
print('all 16 masks CPU PASS')
