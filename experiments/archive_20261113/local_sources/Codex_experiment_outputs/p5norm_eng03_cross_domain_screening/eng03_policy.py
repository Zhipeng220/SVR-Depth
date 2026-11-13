import torch
def build_policy_state(native_state,s0,s1,policy):
 if policy not in ('HOLD','FULL_COMMIT','LAYER1_ONLY'):raise ValueError('unknown policy')
 if set(native_state)!={'features','radius','confidence','rays_map'}:raise ValueError('unexpected state fields')
 if any(len(v)!=4 for v in [s0,s1,native_state['features']]):raise ValueError('expected four layers')
 for a,b,c in zip(s0,s1,native_state['features']):
  if any((x.shape!=a.shape or x.dtype!=a.dtype or x.device!=a.device) for x in (b,c)):raise ValueError('feature metadata')
 for v in list(s0)+list(s1)+list(native_state['features'])+[native_state[k] for k in ['radius','confidence','rays_map']]:
  if not torch.isfinite(v).all():raise ValueError('nonfinite state')
 chosen=s0 if policy=='HOLD' else s1 if policy=='FULL_COMMIT' else [s1[0],*s0[1:]]
 result={k:v.detach().clone() for k,v in native_state.items() if k!='features'};result['features']=[v.detach().clone() for v in chosen]
 pointers=[v.untyped_storage().data_ptr() for v in result['features']];src=[v.untyped_storage().data_ptr() for v in list(s0)+list(s1)+list(native_state['features'])]
 if len(set(pointers))!=4 or set(pointers)&set(src):raise ValueError('feature storage alias')
 return result
