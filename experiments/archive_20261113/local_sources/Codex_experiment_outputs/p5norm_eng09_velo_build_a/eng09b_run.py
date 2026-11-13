import os,sys,subprocess,json,csv
from pathlib import Path
root=Path('/home/svrdepth/SVR-Depth/velodepth/outputs/p5norm_eng09_velo_build_a');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';assert json.loads((root/'cpu_status.json').read_text())['status']=='PASS'
source=root.parent/'p5norm_eng09_velo_raw_stagea/raw_manifest.csv'
with source.open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))[:8]
with (root/'experiment_manifest.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (root/'P5NORM_ENG09_VELO_GPU_LOG.txt').open('x') as log:
 log.write('Parent prefix: frame 0 keyframe + frames 1 and 2 run_step; no threshold\n');log.flush()
 for cmd in [['/usr/lib/wsl/lib/nvidia-smi'],['free','-h']]:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
 with (root/'parent.log').open('x') as f:rc=subprocess.run([sys.executable,str(root/'eng09b_prefix.py')],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 log.write('parent exit '+str(rc)+'\n');log.flush()
 if rc:raise SystemExit('STOP parent failure')
 for policy in ['0000','1000','0100','0010','0001','1111']:
  log.write('\n=== '+policy+' ===\n');log.flush()
  for cmd in [['/usr/lib/wsl/lib/nvidia-smi'],['free','-h']]:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
  with (root/(policy+'.log')).open('x') as f:rc=subprocess.run([sys.executable,str(root/'eng09_velo_rollout.py'),policy],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  log.write('exit '+str(rc)+'\n');log.flush();print(policy,'exit',rc,flush=True)
  if rc:raise SystemExit('STOP '+policy)
print('ALL COMPLETE',flush=True)
