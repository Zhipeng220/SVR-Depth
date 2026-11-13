import datetime,getpass,json,paramiko,shlex,subprocess,sys,time
from pathlib import Path

root=Path.cwd(); scripts=root/'m05x_work/p2a';report=root/'docs/m05x/p2a';tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
datasets=['Bonn','TUM','ScanNet'];ages=[1,4]
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=20);p=None
s=c.open_sftp();remote='/tmp/m05x_p2a_'+tag;s.mkdir(remote)
s.put(str(scripts/'capture_p2a.py'),remote+'/capture_p2a.py');s.put(str(report/'pilot_protocol.json'),remote+'/pilot_protocol.json')
log=[];stopped=False
for dataset in datasets:
 for age in ages:
  name=f'{dataset}_age{age}';target=report/'runs'/name
  if target.exists():raise FileExistsError(target)
  out=remote+'/'+name;cmd='cd ~/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python -u '+shlex.quote(remote+'/capture_p2a.py')+' '+shlex.quote(out)+' '+dataset+' '+str(age)+' '+shlex.quote(remote+'/pilot_protocol.json')
  print('START',name,flush=True);_,stdout,stderr=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=1800);channel=stdout.channel;chunks=[]
  while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
   if channel.recv_ready():chunks.append(channel.recv(65536))
   if channel.recv_stderr_ready():chunks.append(channel.recv_stderr(65536))
   time.sleep(.15)
  rc=channel.recv_exit_status();target.mkdir(parents=True);(target/'capture.log').write_bytes(b''.join(chunks));row={'dataset':dataset,'age':age,'remote_output':out,'remote_command':cmd,'capture_exit':rc}
  if rc:
   print('CAPTURE_FAILED',name,rc,flush=True);row['status']='BLOCKED';log.append(row);stopped=True;break
  for filename in ('capture_metadata.json','evidence_arrays.npz'):
   print('DOWNLOADING',name,filename,flush=True);s.get(out+'/'+filename,str(target/filename))
  verify_dir=target/'verification';v=subprocess.run([sys.executable,str(scripts/'verify_p2a.py'),str(target),'--output-dir',str(verify_dir)],capture_output=True,text=True,encoding='utf-8',errors='replace')
  (target/'verify.log').write_text(v.stdout+'\n'+v.stderr,encoding='utf-8');row.update(verify_exit=v.returncode,status='PASS' if v.returncode==0 else 'INVALID');log.append(row)
  print('VERIFIED',name,row['status'],flush=True)
  (report/'pilot_execution.json').write_text(json.dumps({'run_id':tag,'remote_base':remote,'completed':log},indent=2),encoding='utf-8')
  if v.returncode:stopped=True;break
 if stopped:break
(report/'pilot_execution.json').write_text(json.dumps({'run_id':tag,'remote_base':remote,'completed':log,'stopped_early':stopped},indent=2),encoding='utf-8')
s.close();c.close();print('PILOT_FINISHED',len(log),'stopped_early',stopped,flush=True)
raise SystemExit(1 if stopped else 0)
