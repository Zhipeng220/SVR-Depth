import torch
def build_policy_state(native_state,s0,s1,policy):
 if policy not in ('FULL_COMMIT','NO_L1','NO_L2','NO_L3','NO_L4'):raise ValueError('unknown policy')
 if set(native_state)!={'features','radius','confidence','rays_map'}:raise ValueError('unexpected state fields')
 if any(len(v)!=4 for v in [s0,s1,native_state['features']]):raise ValueError('expected four layers')
 for a,b,c in zip(s0,s1,native_state['features']):
  if any(x.shape!=a.shape or x.dtype!=a.dtype or x.device!=a.device for x in (b,c)):raise ValueError('feature metadata mismatch')
 for x in list(s0)+list(s1)+list(native_state['features'])+[native_state[k] for k in ['radius','confidence','rays_map']]:
  if not torch.isfinite(x).all():raise ValueError('nonfinite')
 removed=int(policy[-1])-1 if policy.startswith('NO_L') else -1
 result={k:v.detach().clone() for k,v in native_state.items() if k!='features'}
 result['features']=[(s0[i] if i==removed else s1[i]).detach().clone() for i in range(4)]
 ptr=[x.untyped_storage().data_ptr() for x in result['features']];src=[x.untyped_storage().data_ptr() for x in list(s0)+list(s1)+list(native_state['features'])]
 if len(set(ptr))!=4 or set(ptr)&set(src):raise ValueError('storage alias')
 return result
