import subprocess,os,sys
from pathlib import Path
root=Path('/home/svrdepth/SVR-Depth/velodepth/outputs/p5norm_eng06_single_layer_h4');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
with (root/'P5NORM_ENG06_GPU_LOG.txt').open('x') as log:
 for policy in ['0000','1000','0100','0010','0001','1111']:
  log.write('\n=== '+policy+' ===\n');log.flush()
  for cmd in [['/usr/lib/wsl/lib/nvidia-smi'],['free','-h']]:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
  with (root/(policy+'.log')).open('x') as child:p=subprocess.run([sys.executable,str(root/'eng06_rollout.py'),policy],env=env,stdout=child,stderr=subprocess.STDOUT)
  log.write('worker exit '+str(p.returncode)+'\n');log.flush();print(policy,'exit',p.returncode,flush=True)
  if p.returncode:raise SystemExit('STOP '+policy)
print('ALL COMPLETE',flush=True)
