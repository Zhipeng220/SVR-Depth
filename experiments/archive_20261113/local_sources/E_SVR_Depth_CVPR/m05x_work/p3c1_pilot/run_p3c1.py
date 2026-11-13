import datetime, getpass, json, paramiko, shlex, subprocess, sys, time
from pathlib import Path

root=Path.cwd(); scripts=root/'m05x_work/p3c1_pilot'; report=root/'docs/m05x/p3c1_pilot'
tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
anchors=json.loads((root/'docs/m05x/p2b/anchor_manifest.json').read_text(encoding='utf8'))['anchors']
lookup={a['anchor_id']:a for a in anchors}
lock=json.loads((report/'pilot_protocol.json').read_text(encoding='utf8'))
work=lock['anchor_ids']; log=[]; stopped=False; reason=''
p=getpass.getpass('SSH password: '); c=paramiko.SSHClient(); c.load_system_host_keys(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=20); p=None
s=c.open_sftp(); remote='/home/svrdepth/SVR-Depth/velodepth/outputs/m05x_p3c1_'+tag; s.mkdir(remote)
for src,name in [(scripts/'capture_p3c1.py','capture_p3c1.py'),(root/'docs/m05x/p2b/sequence_manifest.json','sequence_manifest.json'),(report/'pilot_protocol.json','pilot_protocol.json')]:s.put(str(src),remote+'/'+name)
try:
 for aid in work:
  item=lookup[aid]; target=report/'runs'/aid
  if target.exists():raise FileExistsError(target)
  out=remote+'/'+aid
  args=[remote+'/capture_p3c1.py',out,item['dataset'],str(item['age']),remote+'/sequence_manifest.json',str(item['sequence_index']),remote+'/pilot_protocol.json',aid]
  cmd='cd ~/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python -u '+' '.join(map(shlex.quote,args))
  print('START',aid,flush=True); _,stdout,stderr=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=1800); ch=stdout.channel; chunks=[]
  while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
   if ch.recv_ready():chunks.append(ch.recv(65536))
   if ch.recv_stderr_ready():chunks.append(ch.recv_stderr(65536))
   time.sleep(.1)
  rc=ch.recv_exit_status(); target.mkdir(parents=True); (target/'capture.log').write_bytes(b''.join(chunks))
  row={'anchor_id':aid,'dataset':item['dataset'],'sequence':item['sequence'],'age':item['age'],'remote_output':out,'remote_command':cmd,'capture_exit':rc}
  if rc:
   row['status']='BLOCKED'; log.append(row); stopped=True; reason='capture failed'; print('CAPTURE_FAILED',aid,rc,flush=True); break
  for filename in ('capture_metadata.json','evidence_arrays.npz'):
   print('DOWNLOADING',aid,filename,flush=True); s.get(out+'/'+filename,str(target/filename),prefetch=True,max_concurrent_prefetch_requests=64)
  vd=target/'verification'; v=subprocess.run([sys.executable,str(scripts/'verify_p3c1.py'),str(target),'--output-dir',str(vd)],capture_output=True,text=True,encoding='utf8',errors='replace')
  (target/'verify.log').write_text(v.stdout+'\n'+v.stderr,encoding='utf8'); row.update(verify_exit=v.returncode,status='PASS' if v.returncode==0 else 'INVALID'); log.append(row)
  print('VERIFIED',aid,row['status'],flush=True)
  (report/'execution_progress.json').write_text(json.dumps({'run_id':tag,'remote_base':remote,'completed':log,'stopped_early':False},ensure_ascii=False,indent=2),encoding='utf8')
  if v.returncode:stopped=True; reason='independent invariant or endpoint check failed'; break
except Exception as e:
 stopped=True; reason=repr(e); raise
finally:
 (report/'execution_progress.json').write_text(json.dumps({'run_id':tag,'remote_base':remote,'completed':log,'stopped_early':stopped,'reason':reason},ensure_ascii=False,indent=2),encoding='utf8'); s.close(); c.close()
print('P3C1_FINISHED',len(log),'stopped_early',stopped,flush=True); raise SystemExit(1 if stopped else 0)
