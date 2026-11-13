import base64,csv,getpass,json,paramiko,shlex,time,zlib
from pathlib import Path
root=Path.cwd();work=root/'m05x_work/provenance_followup_20261110';prior=root/'docs/data_audit/provenance_21_20261110_114734';out=root/'docs/data_audit/PROVENANCE_FOLLOWUP';out.mkdir(parents=True,exist_ok=False)
with (prior/'frame_inventory.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
lookup={(r['sequence'],int(r['index']),r['modality']):r for r in rows if r['dataset']=='Bonn'}
configs=[('A: person_tracking2','person_tracking2',100),('A: placing_nonobstructing_box','placing_nonobstructing_box',397),('B: placing_nonobstructing_box','placing_nonobstructing_box',374),('B: removing_nonobstructing_box','removing_nonobstructing_box',53)]
request={'panels':[]}
for label,seq,start in configs:
 p={'label':label,'frames':[]}
 for i in (start,start+1,start+2):
  a=lookup[(seq,i,'rgb')];b=lookup[(seq,i,'depth')];p['frames'].append({'index':i,'rgb_path':a['path'],'depth_path':b['path'],'rgb_timestamp':a['timestamp'],'depth_timestamp':b['timestamp'],'rgb_sha256':a['sha256'],'depth_sha256':b['sha256']})
 request['panels'].append(p)
pw=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
cmd='/home/svrdepth/miniconda3/envs/svrdepth/bin/python -u - '+shlex.quote(json.dumps(request,ensure_ascii=False));stdin,stdout,stderr=c.exec_command(cmd,timeout=600);stdin.write((work/'remote.py').read_text(encoding='utf8'));stdin.channel.shutdown_write();ch=stdout.channel;chunks=[];errs=[]
while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
 if ch.recv_ready():chunks.append(ch.recv(262144))
 if ch.recv_stderr_ready():errs.append(ch.recv_stderr(65536))
 time.sleep(.05)
rc=ch.recv_exit_status();c.close();err=b''.join(errs).decode(errors='replace')
if rc:raise RuntimeError('remote CPU audit failed '+err)
raw=b''.join(chunks).decode('utf8');data=next(x[len('AUDIT_ZLIB_BASE64:'):] for x in raw.splitlines() if x.startswith('AUDIT_ZLIB_BASE64:'));obj=json.loads(zlib.decompress(base64.b64decode(data,validate=True)));png=base64.b64decode(obj.pop('contact_sheet_png_base64'),validate=True);(out/'Bonn_pHash_contact_sheet.png').write_bytes(png);(work/'remote_result.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print('CPU_SCAN_COMPLETE','TUM',[(x['split'],x['target_index'],len(x['neighbors'])) for x in obj['tum']],'BONN_IMAGES',len(obj['bonn_panels']),'SOURCE_CANDIDATES',len(obj['dataset_discovery']))

