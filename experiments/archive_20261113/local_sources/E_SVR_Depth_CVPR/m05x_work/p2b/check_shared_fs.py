import getpass,paramiko
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=20);p=None
_,o,e=c.exec_command("if [ -f /mnt/e/SVR-Depth-CVPR/docs/m05x/p2b/sequence_manifest.json ]; then sha256sum /mnt/e/SVR-Depth-CVPR/docs/m05x/p2b/sequence_manifest.json; else echo NOT_SHARED; fi");print(o.read().decode(),e.read().decode(),flush=True);c.close()
