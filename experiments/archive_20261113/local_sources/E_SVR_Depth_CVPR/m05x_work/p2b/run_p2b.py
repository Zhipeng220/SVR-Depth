import datetime,getpass,json,paramiko,shlex,subprocess,sys,time
from pathlib import Path

root=Path.cwd();scripts=root/'m05x_work/p2b';report=root/'docs/m05x/p2b';tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
anchors=json.loads((report/'anchor_manifest.json').read_text(encoding='utf-8'))['anchors'];log=[];stopped=False
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=20);p=None
s=c.open_sftp();remote='/home/svrdepth/SVR-Depth/velodepth/outputs/m05x_p2b_'+tag;s.mkdir(remote)
s.put(str(scripts/'capture_p2b.py'),remote+'/capture_p2b.py');s.put(str(report/'sequence_manifest.json'),remote+'/sequence_manifest.json')
try:
 for item in anchors:
  name=item['anchor_id'];dataset=item['dataset'];age=item['age'];index=item['sequence_index'];target=report/'runs'/name
  if target.exists():raise FileExistsError(target)
  out=remote+'/'+name;cmd='cd ~/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python -u '+shlex.quote(remote+'/capture_p2b.py')+' '+shlex.quote(out)+' '+dataset+' '+str(age)+' '+shlex.quote(remote+'/sequence_manifest.json')+' '+str(index)
  print('START',len(log)+1,'/',len(anchors),name,flush=True);_,stdout,stderr=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=1800);ch=stdout.channel;chunks=[]
  while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
   if ch.recv_ready():chunks.append(ch.recv(65536))
   if ch.recv_stderr_ready():chunks.append(ch.recv_stderr(65536))
   time.sleep(.1)
  rc=ch.recv_exit_status();target.mkdir(parents=True);(target/'capture.log').write_bytes(b''.join(chunks));row={'anchor_id':name,'dataset':dataset,'sequence':item['sequence'],'age':age,'remote_output':out,'remote_command':cmd,'capture_exit':rc}
  if rc:
   print('CAPTURE_FAILED',name,rc,flush=True);row['status']='BLOCKED';log.append(row);stopped=True;break
  for filename in ('capture_metadata.json','evidence_arrays.npz'):
   print('DOWNLOADING',name,filename,flush=True);s.get(out+'/'+filename,str(target/filename),prefetch=True,max_concurrent_prefetch_requests=64)
  vd=target/'verification';v=subprocess.run([sys.executable,str(scripts/'verify_p2b.py'),str(target),'--output-dir',str(vd)],capture_output=True,text=True,encoding='utf-8',errors='replace')
  (target/'verify.log').write_text(v.stdout+'\n'+v.stderr,encoding='utf-8');row.update(verify_exit=v.returncode,status='PASS' if v.returncode==0 else 'INVALID');log.append(row)
  print('VERIFIED',len(log),'/',len(anchors),name,row['status'],flush=True)
  (report/'execution_progress.json').write_text(json.dumps({'run_id':tag,'remote_base':remote,'completed':log},ensure_ascii=False,indent=2),encoding='utf-8')
  if v.returncode:stopped=True;break
finally:
 (report/'execution_progress.json').write_text(json.dumps({'run_id':tag,'remote_base':remote,'completed':log,'stopped_early':stopped},ensure_ascii=False,indent=2),encoding='utf-8')
 s.close();c.close()
print('P2B_FINISHED',len(log),'stopped_early',stopped,flush=True)
raise SystemExit(1 if stopped else 0)
