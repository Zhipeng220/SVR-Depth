import getpass,paramiko,shlex
from pathlib import Path
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False);p=None
s=c.open_sftp();root=s.normalize('.')+'/SVR-Depth/velodepth/';orig=root+'experiments/p1/gt4a2_y04_v4_bonn_train_v0_2.py'
with s.open(orig,'r') as f:src=f.read().decode()
assert 'Y04-V4-Bonn-Adapt-0.2' in src
src=src.replace('Y04-V4-Bonn-Adapt-0.2','Y04-V5-Preserve-0.1')
marker='    import gt4a2_y04_fair_multiscene as y04\n';assert src.count(marker)==1
src=src.replace(marker,marker+'    import y04_v5_train_cycle as v5_cycle\n    c3.train_cycle = v5_cycle.train_cycle\n')
marker='        "trainer_sha256": sha(Path(__file__)), "scene_review_sha256": freeze_sha,\n';assert src.count(marker)==1
src=src.replace(marker,marker+'        "v5_loss_sha256": sha(ROOT / "experiments/p1/y04_v5_train_cycle.py"), "preserve_lambda": v5_cycle.LAMBDA, "preserve_mask": "all_finite_positive_predictions", "preserve_beta": 1.0,\n')
for path,data in [('experiments/p1/y04_v5_train_cycle.py',Path('v5_train_cycle.py').read_bytes()),('experiments/p1/y04_v5_trainer_v2.py',src.encode())]:
 try:
  with s.open(root+path,'rb') as f:assert f.read()==data
 except FileNotFoundError:
  with s.open(root+path,'wx') as f:f.write(data)
cmd=r'''cd ~/SVR-Depth/velodepth || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate svrdepth || exit 1
unset PYTHONPATH;export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20261025 CUDA_VISIBLE_DEVICES=0
SCRIPT=experiments/p1/y04_v5_trainer_v2.py; python -m py_compile "$SCRIPT" experiments/p1/y04_v5_train_cycle.py || exit 1
BASE="$PWD/outputs/y04_v5_gate_$(date +%Y%m%d_%H%M%S)";mkdir "$BASE" || exit 1;echo "OUTPUT $BASE"
M="$PWD/outputs/y04_v4_manifest_candidate_20261108_190940";S="$HOME/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c";I="$PWD/outputs/gt4a2_c3/gt4a2_c3_initial_head.pt";P="$PWD/experiments/p1/y04_v4_bonn_indomain_protocol_v1.json";C="$PWD/outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/cache"
COMMON=(--mode dev --profile bonn3 --manifest-dir "$M" --snapshot "$S" --init-head "$I" --protocol-file "$P" --deterministic --slots-per-seq 1 --steps 6 --seed 20261025 --cache-root "$C")
run() { local f="$1";shift;"$@" 2>&1 | tee "$f";return ${PIPESTATUS[0]}; }
run "$BASE/a1.log" python "$SCRIPT" "${COMMON[@]}" --run-dir "$BASE/resumed" --stop-after 3 || exit 1
run "$BASE/a2.log" python "$SCRIPT" "${COMMON[@]}" --run-dir "$BASE/resumed" --resume || exit 1
run "$BASE/b.log" python "$SCRIPT" "${COMMON[@]}" --run-dir "$BASE/continuous" || exit 1
python - "$BASE" <<'PY'
import json,sys,csv,math
from pathlib import Path
p=Path(sys.argv[1]);a=json.loads((p/'resumed/completion.json').read_text());b=json.loads((p/'continuous/completion.json').read_text());assert a['head_sha256']==b['head_sha256'];rows=list(csv.DictReader((p/'resumed/training_curve.csv').open(encoding='utf-8-sig')));assert len(rows)==6 and all(math.isfinite(float(x['mean_preserve_loss'])) and int(x['preserve_frames'])==31 for x in rows);print('V5_RESUME_EQUIVALENCE_PASS',a['head_sha256']);(p/'preflight.json').write_text(json.dumps({'status':'PASS','updates':6,'resume_head_sha256':a['head_sha256'],'preserve_lambda':82.71035398147102},indent=2))
PY
'''
_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd),get_pty=True)
for line in o:print(line,end='',flush=True)
print('REMOTE EXIT',o.channel.recv_exit_status(),flush=True);c.close()
