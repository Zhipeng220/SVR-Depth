import numpy as np
from saoof_correspondence import correspondence_mask

h,w=48,64
f=np.zeros((h,w,2),np.float32); b=f.copy()
valid=np.ones((h,w),bool)
m,e=correspondence_mask(f,b,valid)
assert m.all() and np.max(e)==0
f[...,0]=3; b[...,0]=-3
m,e=correspondence_mask(f,b,valid)
assert not m[:,:3].any() and m[:,3:].all()
f[...,0]=.5; b[...,0]=-.5
m,e=correspondence_mask(f,b,valid)
assert not m[:,0].any() and m[:,1:].all() and np.max(e[m])==0
f[...,0]=3; b[...,0]=-3
b[10:20,20:30,0]=5
m,e=correspondence_mask(f,b,valid)
assert not m[10:20,20:30].any()
valid[25:30,30:35]=False
m,e=correspondence_mask(f,b,valid)
assert not m[25:30,30:35].any()
print('PASS: zero flow, integer/subpixel translation, inconsistent flow, bounds and valid-depth mask')
