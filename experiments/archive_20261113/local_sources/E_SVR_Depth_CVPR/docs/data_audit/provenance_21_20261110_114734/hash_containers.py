import getpass,json,paramiko
from pathlib import Path
root=Path.cwd();out=root/'docs/data_audit/provenance_21_20261110_114734';pw=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
cmd='sha256sum /home/svrdepth/datasets/velodepth_evaluation/TUM.hdf5 /home/svrdepth/datasets/velodepth_evaluation/ScanNet.hdf5';i,o,e=c.exec_command(cmd,timeout=600);raw=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();c.close()
if rc:raise RuntimeError(err)
items=[{'path':line.split(None,1)[1],'sha256':line.split(None,1)[0]} for line in raw.splitlines() if line.strip()]
(out/'container_hashes.json').write_text(json.dumps({'command':cmd,'files':items},ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('HDF5_CONTAINER_HASHED',len(items))

