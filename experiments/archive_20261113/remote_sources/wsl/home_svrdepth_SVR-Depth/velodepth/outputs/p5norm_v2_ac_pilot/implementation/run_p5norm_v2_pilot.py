"""Serial A then C per preselected sequence; stop at any failed worker."""
import sys,csv,subprocess,os,json,time
from pathlib import Path
root=Path('/home/svrdepth/SVR-Depth/velodepth/outputs/p5norm_v2_ac_pilot');sel=list(csv.DictReader((root/'sequence_selection/pilot_sequence_selection.csv').open()));worker=Path(__file__).with_name('p5norm_v2_sequence_worker.py')
for r in sel:
 domain,seq=r['dataset'],r['sequence_id'];safe=Path(seq).name
 for strategy in ['A','C']:
  out=root/domain/safe/strategy
  if out.exists():raise SystemExit('STOP existing result: '+str(out))
  out.parent.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';log=out.parent/(strategy+'.log')
  with log.open('x') as f:p=subprocess.Popen([sys.executable,str(worker),domain,seq,strategy,str(out)],env=env,stdout=f,stderr=subprocess.STDOUT);rc=p.wait()
  if rc:raise SystemExit('STOP worker failed: '+str(log))
  print('COMPLETE',domain,seq,strategy,flush=True)
