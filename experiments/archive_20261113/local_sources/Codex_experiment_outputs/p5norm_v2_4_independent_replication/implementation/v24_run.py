from pathlib import Path
import os,sys,csv,subprocess,json
root=Path('/home/svrdepth/SVR-Depth/velodepth/outputs/p5norm_v2_4_independent_replication');seqs=list(csv.DictReader((root/'P5NORM_V2_4_FROZEN_SEQUENCE_LIST.csv').open()));anchors=list(csv.DictReader((root/'P5NORM_V2_4_FROZEN_ANCHORS.csv').open()));smoke=[next(x for x in seqs if x['dataset']==d) for d in ['Bonn','TUM','ScanNet']];ordered=smoke+[x for x in seqs if x not in smoke];worker=Path('/tmp/p5norm_v24_worker.py');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
for j,s in enumerate(ordered):
 d=s['dataset'];seq=s['sequence_id'];folder=root/'runs'/d/Path(seq).name;folder.mkdir(parents=True,exist_ok=True);aa=[a for a in anchors if (a['dataset'],a['sequence_id'])==(d,seq)]
 for mode,idx in [('hold',0)]+[('branch',int(a['anchor_idx'])) for a in aa]:
  dest=folder/('hold' if mode=='hold' else 'anchor_'+str(idx));log=folder/('hold.log' if mode=='hold' else 'anchor_'+str(idx)+'.log')
  if dest.exists():
   status=dest/'status.json'
   if status.exists() and json.loads(status.read_text())['status']=='PASS':print('already completed',dest,flush=True);continue
   raise SystemExit('STOP existing incomplete run '+str(dest))
  with log.open('x') as f:rc=subprocess.run([sys.executable,str(worker),d,seq,mode,str(idx),str(dest)],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  if rc:raise SystemExit('STOP worker failed: '+str(log))
  print('PASS',d,seq,mode,idx,flush=True)
  if mode=='branch' and json.loads((dest/'status.json').read_text()).get('parent_functional_mismatch'):print('PARENT_FUNCTIONAL_MISMATCH exclude from analysis',d,seq,idx,flush=True)
 if j==2:print('THREE DOMAIN FUNCTIONAL RUNS COMPLETE',flush=True)
print('ALL RUNS COMPLETE',flush=True)
