import os,sys,subprocess
from pathlib import Path
root=Path('/home/svrdepth/SVR-Depth/velodepth/outputs/p5norm_eng09_velo_build_a');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
with (root/'P5NORM_ENG09_VELO_GPU_LOG.txt').open('a') as log:
 log.write('\nContinue five unrun branches; no HOLD retry; original total budget 30 run_step\n');log.flush()
 for policy in ['1000','0100','0010','0001','1111']:
  log.write('\n=== '+policy+' ===\n');log.flush()
  for cmd in [['/usr/lib/wsl/lib/nvidia-smi'],['free','-h']]:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
  with (root/(policy+'.log')).open('x') as f:rc=subprocess.run([sys.executable,str(root/'eng09_velo_rollout_fixed.py'),policy],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  log.write('exit '+str(rc)+'\n');log.flush();print(policy,'exit',rc,flush=True)
  if rc:raise SystemExit('STOP '+policy)
print('FIVE REMAINING COMPLETE',flush=True)
