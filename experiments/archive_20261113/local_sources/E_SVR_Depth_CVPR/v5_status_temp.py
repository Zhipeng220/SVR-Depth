import getpass,paramiko,shlex
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False);p=None
cmd=r'''cd ~/SVR-Depth/velodepth || exit 1
R=outputs/y04_v5_preserve_seed20261025_20261109_112607
ls -lh "$R/recovery.pt" "$R/training_curve.csv"
~/miniconda3/envs/svrdepth/bin/python - "$R" <<'PY'
import sys,csv,torch,math
from pathlib import Path
p=Path(sys.argv[1]);ck=torch.load(p/'recovery.pt',map_location='cpu',weights_only=False);r=list(csv.DictReader((p/'training_curve.csv').open(encoding='utf-8-sig')));print('RECOVERY_STEP',ck['completed_updates'],'CURVE_ROWS',len(r));print('LATEST',r[-1]);print('ALL_PRESERVE_FINITE',all(math.isfinite(float(x['mean_preserve_loss'])) for x in r))
PY'''
_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd));print(o.read().decode());print(e.read().decode());c.close()
