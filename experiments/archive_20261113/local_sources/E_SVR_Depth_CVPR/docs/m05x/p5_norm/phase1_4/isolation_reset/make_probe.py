from pathlib import Path
p=Path('docs/m05x/p5_norm/phase1_3/runtime_server_probe.py');s=p.read_text()
s=s.replace('for rep in range(2):','for rep in range(1):').replace("for group in ['G0','G1','G2','G3']:","for group in ['G1']:").replace('for i in range(1,8):','for i in range(1,2):')
marker='    a=m05.clone_state(parent);b=m05.clone_state(parent)'
insert='''    tri=[m05.clone_state(parent) for _ in range(3)]
    for src in range(3):
     for k,x in tensors(tri[src]).items():
      ptrs=[tensors(z)[k].untyped_storage().data_ptr() for z in tri]
      if len(set(ptrs))!=3:raise RuntimeError('three-way shared storage '+k)
      before_others=[tensorhash(tensors(tri[j])[k]) for j in range(3) if j!=src]
      old=x.reshape(-1)[0].item();x.reshape(-1)[0]=old+1
      assert before_others==[tensorhash(tensors(tri[j])[k]) for j in range(3) if j!=src]
      x.reshape(-1)[0]=old
      audit.append({'group':['G1','G2','G3'][src],'state_field':k,'pointers':ptrs,'mutation_isolated':True})
'''
assert marker in s;s=s.replace(marker,insert+marker)
Path('docs/m05x/p5_norm/phase1_4/isolation_reset/probe.py').write_text(s)
