import subprocess,os,sys
from pathlib import Path
root=Path('/home/svrdepth/SVR-Depth/velodepth/outputs/p5norm_eng08_temporal_position_robustness');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
with (root/'P5NORM_ENG08_GPU_LOG.txt').open('x') as log:
 log.write('Common HOLD prefix: 2 run_step\n');log.flush()
 for cmd in [['/usr/lib/wsl/lib/nvidia-smi'],['free','-h']]:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
 with (root/'prefix.log').open('x') as f:rc=subprocess.run([sys.executable,str(root/'eng08_prefix.py')],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 log.write('prefix exit '+str(rc)+'\n');log.flush()
 if rc:raise SystemExit('STOP prefix')
 for pos in ['P2','P3']:
  for policy in ['0000','1000','0100','0010','0001','1111']:
   log.write('\n=== '+pos+' '+policy+' ===\n');log.flush()
   for cmd in [['/usr/lib/wsl/lib/nvidia-smi'],['free','-h']]:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
   with (root/(pos+'_'+policy+'.log')).open('x') as f:rc=subprocess.run([sys.executable,str(root/'eng08_rollout.py'),policy,pos],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
   log.write('exit '+str(rc)+'\n');log.flush();print(pos,policy,'exit',rc,flush=True)
   if rc:raise SystemExit('STOP '+pos+' '+policy)
print('ALL COMPLETE',flush=True)
