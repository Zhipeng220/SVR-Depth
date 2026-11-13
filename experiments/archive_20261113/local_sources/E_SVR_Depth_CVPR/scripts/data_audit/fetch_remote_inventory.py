import getpass,json,paramiko
from pathlib import Path
root=Path.cwd();script=(root/'scripts/data_audit/remote_inventory.py').read_text(encoding='utf8');out=root/'docs/data_audit';out.mkdir(parents=True,exist_ok=True)
pw=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
stdin,stdout,stderr=c.exec_command('/home/svrdepth/miniconda3/envs/svrdepth/bin/python -',timeout=1800);stdin.write(script);stdin.channel.shutdown_write();data=stdout.read();err=stderr.read();rc=stdout.channel.recv_exit_status();c.close()
if rc:raise RuntimeError('remote read-only scan failed '+err.decode(errors='replace'))
obj=json.loads(data);(out/'remote_inventory.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('DATASETS', {k:len(v['items']) for k,v in obj['datasets'].items()},'BONN',len(obj['bonn']),'ERRORS',obj['errors'])
