import getpass,hashlib,json,paramiko,shlex,subprocess,sys,time
from pathlib import Path
root=Path.cwd();out=root/'outputs/P4A_CrossDomain_Replication_20261110_110108';p=out/'preflight';pre=json.loads((p/'P4A_CrossDomain_Preflight.json').read_text());tum=json.loads((out/'TUM/audit/independent_validation.json').read_text());initial=json.loads((out/'execution_status.json').read_text());
if pre['status']!='PASS' or tum['status']!='PASS':raise RuntimeError('preflight or TUM verification gate blocked')
(out/'TUM/logs/independent_validation_after_fix.json').write_text(json.dumps(tum,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
pw=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
s=c.open_sftp();remote='/home/svrdepth/SVR-Depth/velodepth/outputs/P4A_CrossDomain_Replication_20261110_110108';script=p/'cross_domain_capture.py'
with s.open(remote+'/cross_domain_capture.py','rb') as f:remote_hash=hashlib.sha256(f.read()).hexdigest()
if remote_hash!=hashlib.sha256(script.read_bytes()).hexdigest():raise RuntimeError('capture script changed')
target=out/'ScanNet';row={'domain':'ScanNet','anchor':'ScanNet_00_age4'};error=''
try:
 if target.exists():raise FileExistsError(target)
 for name in ['capture','metrics','equivalence','logs']:(target/name).mkdir(parents=True,exist_ok=True)
 local=pre['local_history']['ScanNet'];args=[remote+'/cross_domain_capture.py','ScanNet','ScanNet_00_age4',local['metadata'],local['npz'],remote+'/ScanNet'];cmd='cd ~/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python -u '+' '.join(map(shlex.quote,args));row['remote_command']=cmd;print('START ScanNet',flush=True)
 _,stdout,stderr=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=1800);ch=stdout.channel;chunks=[]
 while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
  if ch.recv_ready():chunks.append(ch.recv(65536))
  if ch.recv_stderr_ready():chunks.append(ch.recv_stderr(65536))
  time.sleep(.1)
 rc=ch.recv_exit_status();(target/'logs/remote_capture.log').write_bytes(b''.join(chunks));row['capture_exit']=rc
 if rc:raise RuntimeError('ScanNet capture failed '+str(rc))
 for src,dst in [('capture.npz','capture/capture.npz'),('result.json','metrics/result.json')]:print('DOWNLOAD',src,flush=True);s.get(remote+'/ScanNet/'+src,str(target/dst),prefetch=True,max_concurrent_prefetch_requests=64)
 result=json.loads((target/'metrics/result.json').read_text());(target/'equivalence/baseline_diagnostic.json').write_text(json.dumps(result['baseline_vs_diagnostic'],indent=2)+'\n')
 v=subprocess.run([sys.executable,str(p/'verify_capture.py'),str(target),'ScanNet_00_age4'],capture_output=True,text=True,encoding='utf8',errors='replace');(target/'logs/independent_verify.log').write_text(v.stdout+'\n'+v.stderr,encoding='utf8');row['verify_exit']=v.returncode;row['status']='PASS' if v.returncode==0 and result['pass'] else 'FAIL'
 if row['status']!='PASS':raise RuntimeError('ScanNet independent verification failed')
 print('VERIFIED ScanNet PASS',flush=True)
except Exception as e:error=repr(e);row['status']='FAIL';print('FAIL',error,flush=True)
finally:
 (out/'execution_status.json').write_text(json.dumps({'initial_attempt':initial,'tum_reverification':'PASS after correcting verifier array-key spelling; original failed logs preserved','scannet':row,'error':error},indent=2,ensure_ascii=False)+'\n',encoding='utf8');s.close();c.close()
raise SystemExit(1 if error else 0)
