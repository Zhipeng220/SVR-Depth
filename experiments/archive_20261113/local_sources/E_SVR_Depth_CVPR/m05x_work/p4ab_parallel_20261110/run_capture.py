import getpass,json,paramiko,shlex,subprocess,sys,time
from pathlib import Path
root=Path.cwd();work=root/'m05x_work/p4ab_parallel_20261110';lock=json.loads((work/'protocol_lock.json').read_text(encoding='utf8'));selection=json.loads((work/'capture_selection.json').read_text(encoding='utf8'))
if selection['protocol_lock_sha256']!=__import__('hashlib').sha256((work/'protocol_lock.json').read_bytes()).hexdigest():raise RuntimeError('locked protocol mismatch')
verify_src=root/'outputs/P4A_CrossDomain_Replication_20261110_110108/preflight/verify_capture.py'
source=verify_src.read_text(encoding='utf8');old="out=p/'audit';out.mkdir(parents=True,exist_ok=True)";new="out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)"
if source.count(old)!=1:raise RuntimeError('verify adaptation mismatch')
(work/'verify_capture.py').write_text(source.replace(old,new),encoding='utf8')
records=[];reused=root/'outputs/P4A_CrossDomain_Replication_20261110_110108'
for ds,aid in [('TUM','TUM_00_age4'),('ScanNet','ScanNet_00_age4')]:
 path=reused/ds;v=subprocess.run([sys.executable,str(work/'verify_capture.py'),str(path),aid,str(work/'reverified'/aid)],capture_output=True,text=True,encoding='utf8',errors='replace')
 (work/'reverified'/aid/'verification.log').write_text(v.stdout+'\n'+v.stderr,encoding='utf8')
 if v.returncode:raise RuntimeError('reused capture failed verification '+aid)
 records.append({'anchor_id':aid,'dataset':ds,'status':'PASS','source':'reused','path':str(path)})
print('REUSED_VERIFIED',len(records),flush=True)
pw=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
s=c.open_sftp();remote='/home/svrdepth/SVR-Depth/velodepth/outputs/p4ab_parallel_20261110';s.mkdir(remote);s.put(str(work/'capture.py'),remote+'/capture.py');error=''
try:
 for row in selection['records']:
  aid=row['anchor_id'];ds=row['dataset']
  if aid in ['TUM_00_age4','ScanNet_00_age4']:continue
  target=work/'captures'/aid
  if target.exists():raise FileExistsError(target)
  for x in ['capture','metrics','logs']:(target/x).mkdir(parents=True,exist_ok=True)
  args=[remote+'/capture.py',ds,aid,row['capture_metadata.json'],row['evidence_arrays.npz'],remote+'/'+aid];cmd='cd ~/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python -u '+' '.join(map(shlex.quote,args));print('START',len(records)+1,'/39',aid,flush=True)
  _,stdout,stderr=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=1800);ch=stdout.channel;chunks=[]
  while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
   if ch.recv_ready():chunks.append(ch.recv(65536))
   if ch.recv_stderr_ready():chunks.append(ch.recv_stderr(65536))
   time.sleep(.1)
  rc=ch.recv_exit_status();(target/'logs/capture.log').write_bytes(b''.join(chunks));rec={'anchor_id':aid,'dataset':ds,'capture_exit':rc,'path':str(target),'source':'fresh','remote_command':cmd};records.append(rec)
  if rc:rec['status']='FAIL';error='capture failed '+aid;break
  for src,dst in [('capture.npz','capture/capture.npz'),('result.json','metrics/result.json')]:
   print('DOWNLOAD',aid,src,flush=True);s.get(remote+'/'+aid+'/'+src,str(target/dst),prefetch=True,max_concurrent_prefetch_requests=64)
  v=subprocess.run([sys.executable,str(work/'verify_capture.py'),str(target),aid,str(target/'audit')],capture_output=True,text=True,encoding='utf8',errors='replace');(target/'logs/verify.log').write_text(v.stdout+'\n'+v.stderr,encoding='utf8');rec['verify_exit']=v.returncode;rec['status']='PASS' if v.returncode==0 else 'FAIL'
  (work/'capture_progress.json').write_text(json.dumps({'records':records,'error':''},indent=2,ensure_ascii=False)+'\n',encoding='utf8')
  print('VERIFIED',len(records),'/39',aid,rec['status'],flush=True)
  if v.returncode:error='independent verification failed '+aid;break
except Exception as e:error=repr(e);raise
finally:
 (work/'capture_progress.json').write_text(json.dumps({'records':records,'error':error},indent=2,ensure_ascii=False)+'\n',encoding='utf8');s.close();c.close()
print('CAPTURE_FINISHED',len(records),'error',error,flush=True);raise SystemExit(1 if error else 0)

