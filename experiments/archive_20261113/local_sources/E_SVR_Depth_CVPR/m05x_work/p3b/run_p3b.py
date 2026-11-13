import datetime,getpass,json,paramiko,shlex,subprocess,sys,time
from pathlib import Path
root=Path.cwd();scripts=root/'m05x_work/p3b';report=root/'docs/m05x/p3b';tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
anchors=json.loads((root/'docs/m05x/p2b/anchor_manifest.json').read_text(encoding='utf8'))['anchors'];lookup={a['anchor_id']:a for a in anchors};lock=json.loads((report/'protocol_lock.json').read_text(encoding='utf8'));pilot=lock['pilot_anchor_ids'];formal=[a['anchor_id'] for a in anchors if a['anchor_id'] not in pilot];work=[('pilot',x) for x in pilot]+[('formal',x) for x in formal]
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=20);p=None
s=c.open_sftp();remote='/home/svrdepth/SVR-Depth/velodepth/outputs/m05x_p3b_'+tag;s.mkdir(remote)
for src,name in [(scripts/'capture_p3b.py','capture_p3b.py'),(root/'docs/m05x/p2b/sequence_manifest.json','sequence_manifest.json'),(report/'protocol_lock.json','protocol_lock.json')]:s.put(str(src),remote+'/'+name)
log=[];stopped=False;reason=''
try:
 for phase,aid in work:
  if phase=='formal' and (len(log)<3 or any(x['status']!='PASS' for x in log[:3])):raise RuntimeError('pilot gate not passed')
  item=lookup[aid];dataset=item['dataset'];age=item['age'];index=item['sequence_index'];target=report/'runs'/aid
  if target.exists():raise FileExistsError(target)
  out=remote+'/'+aid;cmd='cd ~/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python -u '+' '.join(map(shlex.quote,[remote+'/capture_p3b.py',out,dataset,str(age),remote+'/sequence_manifest.json',str(index),remote+'/protocol_lock.json',aid,phase]))
  print('START',len(log)+1,'/',len(work),phase,aid,flush=True);_,stdout,stderr=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=1800);ch=stdout.channel;chunks=[]
  while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
   if ch.recv_ready():chunks.append(ch.recv(65536))
   if ch.recv_stderr_ready():chunks.append(ch.recv_stderr(65536))
   time.sleep(.1)
  rc=ch.recv_exit_status();target.mkdir(parents=True);(target/'capture.log').write_bytes(b''.join(chunks));row={'anchor_id':aid,'phase':phase,'dataset':dataset,'sequence':item['sequence'],'age':age,'remote_output':out,'remote_command':cmd,'capture_exit':rc}
  if rc:row['status']='BLOCKED';log.append(row);stopped=True;reason='capture failed';print('CAPTURE_FAILED',aid,rc,flush=True);break
  for filename in ('capture_metadata.json','evidence_arrays.npz'):
   print('DOWNLOADING',aid,filename,flush=True);s.get(out+'/'+filename,str(target/filename),prefetch=True,max_concurrent_prefetch_requests=64)
  vd=target/'verification';v=subprocess.run([sys.executable,str(scripts/'verify_p3b.py'),str(target),'--output-dir',str(vd)],capture_output=True,text=True,encoding='utf8',errors='replace');(target/'verify.log').write_text(v.stdout+'\n'+v.stderr,encoding='utf8');row.update(verify_exit=v.returncode,status='PASS' if v.returncode==0 else 'INVALID');log.append(row)
  print('VERIFIED',len(log),'/',len(work),phase,aid,row['status'],flush=True)
  (report/'execution_progress.json').write_text(json.dumps({'run_id':tag,'remote_base':remote,'completed':log,'stopped_early':False},ensure_ascii=False,indent=2),encoding='utf8')
  if v.returncode:stopped=True;reason='independent invariant or endpoint check failed';break
except Exception as e:stopped=True;reason=repr(e);raise
finally:
 (report/'execution_progress.json').write_text(json.dumps({'run_id':tag,'remote_base':remote,'completed':log,'stopped_early':stopped,'reason':reason},ensure_ascii=False,indent=2),encoding='utf8');s.close();c.close()
print('P3B_FINISHED',len(log),'pilot_passed',len(log)>=3 and all(x['status']=='PASS' for x in log[:3]),'stopped_early',stopped,flush=True);raise SystemExit(1 if stopped else 0)
