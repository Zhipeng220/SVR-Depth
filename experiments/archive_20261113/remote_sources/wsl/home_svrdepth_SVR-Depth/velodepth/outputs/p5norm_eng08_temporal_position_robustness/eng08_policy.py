import torch
def build_policy_state(native_state,s0,s1,policy):
 if len(policy)!=4 or set(policy)-{'0','1'}:raise ValueError('mask must have four binary characters')
 if set(native_state)!={'features','radius','confidence','rays_map'}:raise ValueError('unknown state fields')
 if any(len(v)!=4 for v in [s0,s1,native_state['features']]):raise ValueError('expected four layers')
 for a,b,c in zip(s0,s1,native_state['features']):
  if any(x.shape!=a.shape or x.dtype!=a.dtype or x.device!=a.device for x in (b,c)):raise ValueError('metadata')
 for x in list(s0)+list(s1)+list(native_state['features'])+[native_state[k] for k in ['radius','confidence','rays_map']]:
  if not torch.isfinite(x).all():raise ValueError('nonfinite')
 result={k:v.detach().clone() for k,v in native_state.items() if k!='features'};result['features']=[(s1[i] if policy[i]=='1' else s0[i]).detach().clone() for i in range(4)]
 ptr=[x.untyped_storage().data_ptr() for x in result['features']];src=[x.untyped_storage().data_ptr() for x in list(s0)+list(s1)+list(native_state['features'])]
 if len(set(ptr))!=4 or set(ptr)&set(src):raise ValueError('feature alias')
 for k in ['radius','confidence','rays_map']:
  if not torch.equal(result[k],native_state[k]) or result[k].untyped_storage().data_ptr()==native_state[k].untyped_storage().data_ptr():raise ValueError('nonfeature alias')
 return result
