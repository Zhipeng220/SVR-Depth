import datetime,getpass,os,paramiko,shlex,sys
from pathlib import Path
p=getpass.getpass('SSH password: '); c=paramiko.SSHClient(); c.load_system_host_keys(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=15);p=None
tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote_script='/tmp/svrdepth_p15_capture_'+tag+'.py';remote_out='/tmp/svrdepth_p15_evidence_'+tag
local=Path('docs/m05x/p1_5');s=c.open_sftp();s.put(str(local/'capture_invariants.py'),remote_script)
cmd='cd ~/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python -u '+shlex.quote(remote_script)+' '+shlex.quote(remote_out)+' '+shlex.quote(sys.argv[1] if len(sys.argv)>1 else '1')
print('REMOTE_COMMAND',cmd,flush=True)
_,stdout,stderr=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=900)
channel=stdout.channel; chunks=[]
while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
    import time
    if channel.recv_ready():
        b=channel.recv(65536); print(b.decode(errors='replace'),end='',flush=True);chunks.append(b)
    if channel.recv_stderr_ready():
        b=channel.recv_stderr(65536); print(b.decode(errors='replace'),end='',flush=True);chunks.append(b)
    time.sleep(.15)
rc=channel.recv_exit_status();(local/('capture_execution_'+tag+'.log')).write_bytes(b''.join(chunks)); print('REMOTE_EXIT',rc,flush=True)
if rc==0:
    target=local/('evidence_new_run_'+tag);target.mkdir(exist_ok=False)
    for name in ['capture_metadata.json','evidence_arrays.npz']:
        print('DOWNLOADING',name,flush=True);s.get(remote_out+'/'+name,str(target/name))
    (target/'remote_path.txt').write_text(remote_out+'\n',encoding='utf-8')
s.close();c.close()
raise SystemExit(rc)
