import torch
from eng01_policy import layer1_writeback

def test():
 s0=[torch.full((2,3),float(i)) for i in range(4)];s1=[x+10 for x in s0];state={'features':s1,'radius':torch.tensor([2.]),'confidence':torch.tensor([3.]),'rays_map':torch.tensor([4.])}
 original=[x.clone() for x in s0+s1];w=layer1_writeback(state,s0,s1)
 assert torch.equal(w['features'][0],s1[0])
 assert all(torch.equal(w['features'][i],s0[i]) for i in range(1,4))
 for i,x in enumerate(w['features']):
  assert all(x.untyped_storage().data_ptr()!=a.untyped_storage().data_ptr() for a in s0+s1)
  assert all(i==j or x.untyped_storage().data_ptr()!=y.untyped_storage().data_ptr() for j,y in enumerate(w['features']))
 for k in ('radius','confidence','rays_map'):assert torch.equal(state[k],w[k]) and state[k].untyped_storage().data_ptr()!=w[k].untyped_storage().data_ptr()
 w['features'][0].add_(100);w['radius'].add_(100)
 assert all(torch.equal(a,b) for a,b in zip(s0+s1,original)) and state['radius'].item()==2
 try:layer1_writeback(state,s0[:3],s1)
 except ValueError:pass
 else:raise AssertionError('invalid layer count accepted')
 print('CPU unit tests PASS: layer selection, clone isolation, nonfeature state, invalid structure')
if __name__=='__main__':test()
