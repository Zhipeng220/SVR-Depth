import getpass, paramiko, shlex
from pathlib import Path
p=getpass.getpass('SSH password: ')
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=15);p=None
cmd=r'''cd ~/SVR-Depth/velodepth || exit 1
echo '=== CURRENT STATUS ==='
date -Is
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
/usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader
/home/svrdepth/miniconda3/envs/svrdepth/bin/python -c "import torch, cv2, numpy; print('torch',torch.__version__,'cuda_available',torch.cuda.is_available(),'cuda_version',torch.version.cuda)"
echo '=== ENVIRONMENTS ==='
ls -d ~/miniconda3/envs/svrdepth ~/anaconda3/envs/svrdepth ~/miniforge3/envs/svrdepth 2>/dev/null
echo '=== DATA ==='
ls -l outputs/y04_v4_manifest_candidate_20261108_190940/moving_obstructing_box_pairs_v4_candidate.csv outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/final.pt outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt
echo '=== RELEVANT SOURCE ==='
sha256sum experiments/p1/cvpr_m05_formal.py experiments/p1/gt4a2_exp1_train_eval.py experiments/p1/gt2_fixed_clip_repair_train.py
'''
_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd)); data=o.read().decode(errors='replace')+e.read().decode(errors='replace');print(data);Path('docs/m05x/p1_5/remote_probe.txt').write_text(data,encoding='utf-8')
s=c.open_sftp(); s.get('/home/svrdepth/SVR-Depth/velodepth/experiments/p1/cvpr_m05_formal.py','docs/m05x/p1_5/cvpr_m05_formal_server_snapshot.py.txt'); s.close();c.close()
