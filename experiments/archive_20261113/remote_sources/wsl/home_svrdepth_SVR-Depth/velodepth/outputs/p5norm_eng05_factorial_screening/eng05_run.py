import subprocess,os,sys
from pathlib import Path
root=Path('/home/svrdepth/SVR-Depth/velodepth/outputs/p5norm_eng05_factorial_screening');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['ENG02_STEPS']='8'
with (root/'P5NORM_ENG05_GPU_LOG.txt').open('x') as log:
 for policy in ['0001','0010','0011','0100','0101','0110','1001','1010','1100']:
  log.write('\n=== '+policy+' ===\n');log.flush()
  for cmd in [['/usr/lib/wsl/lib/nvidia-smi'],['free','-h']]:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
  log.flush()
  with (root/(policy+'.log')).open('x') as child:p=subprocess.run([sys.executable,str(root/'eng05_rollout.py'),policy],env=env,stdout=child,stderr=subprocess.STDOUT)
  log.write('worker exit '+str(p.returncode)+'\n');log.flush();print(policy,'exit',p.returncode,flush=True)
  if p.returncode:raise SystemExit('STOP '+policy)
print('ALL COMPLETE',flush=True)
