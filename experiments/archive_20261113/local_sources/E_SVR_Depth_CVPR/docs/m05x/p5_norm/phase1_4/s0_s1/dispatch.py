import paramiko,getpass
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=getpass.getpass(),look_for_keys=False,allow_agent=False,timeout=12)
s=c.open_sftp();p='/tmp/svrdepth_p5norm_1_4_s0_probe';s.mkdir(p);s.put('docs/m05x/p5_norm/phase1_4/s0_s1/probe.py',p+'/probe.py');s.put('docs/m05x/p5_norm/phase1_1/model_recovery/fold_00.json',p+'/fold.json');s.close();_,o,e=c.exec_command('cd /home/svrdepth/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python -u /tmp/svrdepth_p5norm_1_4_s0_probe/probe.py /tmp/svrdepth_p5norm_1_4_s0_probe');print(o.read().decode());print(e.read().decode());s=c.open_sftp()
try:s.get(p+'/events.json','docs/m05x/p5_norm/phase1_4/s0_s1/events.json')
except FileNotFoundError:print('EVENTS_MISSING')
s.close();c.close()
