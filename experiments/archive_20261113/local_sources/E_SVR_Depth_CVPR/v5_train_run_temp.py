import getpass,paramiko,shlex
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False);p=None
cmd=r'''cd ~/SVR-Depth/velodepth || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate svrdepth || exit 1
unset PYTHONPATH;export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20261025 CUDA_VISIBLE_DEVICES=0
python - <<'PY' || exit 1
import json,hashlib
from pathlib import Path
root=Path.cwd();g=root/'outputs/y04_v5_gate_20261109_112120';assert json.loads((g/'preflight.json').read_text())['status']=='PASS';i=json.loads((g/'resumed/run_identity.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(root/'experiments/p1/y04_v5_trainer_v2.py')==i['trainer_sha256'];assert sha(root/'experiments/p1/y04_v5_train_cycle.py')==i['v5_loss_sha256'];print('V5_GATE_IDENTITY_VERIFIED',flush=True)
PY
RUN="$PWD/outputs/y04_v5_preserve_seed20261025_$(date +%Y%m%d_%H%M%S)";echo "RUN_DIR $RUN"
python experiments/p1/y04_v5_trainer_v2.py --mode adapt --profile bonn13 --manifest-dir "$PWD/outputs/y04_v4_manifest_candidate_20261108_190940" --snapshot "$HOME/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c" --init-head "$PWD/outputs/gt4a2_c3/gt4a2_c3_initial_head.pt" --protocol-file "$PWD/experiments/p1/y04_v4_bonn_indomain_protocol_v1.json" --deterministic --slots-per-seq 8 --steps 1000 --seed 20261025 --save-every 25 --cache-root "$PWD/outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/cache" --run-dir "$RUN" 2>&1 | tee "$RUN.log"
rc=${PIPESTATUS[0]};echo "V5_TRAIN_EXIT $rc";exit "$rc"'''
_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd),get_pty=True)
for line in o:print(line,end='',flush=True)
print('REMOTE EXIT',o.channel.recv_exit_status(),flush=True);c.close()
