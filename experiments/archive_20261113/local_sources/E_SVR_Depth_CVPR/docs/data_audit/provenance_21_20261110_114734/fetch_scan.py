import base64,getpass,json,paramiko,shlex,time,zlib
from pathlib import Path
root=Path.cwd();out=root/'docs/data_audit/provenance_21_20261110_114734';selection=json.loads((out/'selected_sequences.json').read_text(encoding='utf8'))['sequences']
pw=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
cmd='/home/svrdepth/miniconda3/envs/svrdepth/bin/python -u - '+shlex.quote(json.dumps(selection,ensure_ascii=False));stdin,stdout,stderr=c.exec_command(cmd,timeout=3600);stdin.write((out/'remote_scan.py').read_text(encoding='utf8'));stdin.channel.shutdown_write();ch=stdout.channel;chunks=[];errors=[]
while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
 if ch.recv_ready():chunks.append(ch.recv(262144))
 if ch.recv_stderr_ready():errors.append(ch.recv_stderr(65536))
 time.sleep(.05)
rc=ch.recv_exit_status();c.close();text=b''.join(chunks).decode('utf8',errors='replace');err=b''.join(errors).decode('utf8',errors='replace');(out/'remote_scan.log').write_text('exit='+str(rc)+'\n'+err,encoding='utf8')
if rc:raise RuntimeError('remote scan failed '+err[-3000:])
line=next((x[len('AUDIT_ZLIB_BASE64:'):] for x in text.splitlines() if x.startswith('AUDIT_ZLIB_BASE64:')),None)
if not line:raise RuntimeError('missing remote inventory')
raw=zlib.decompress(base64.b64decode(line,validate=True));obj=json.loads(raw);(out/'remote_inventory.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('SCANNED',len(obj['sequences']),'sequences',len(obj['frames']),'frames',len(obj['errors']),'errors',flush=True)

