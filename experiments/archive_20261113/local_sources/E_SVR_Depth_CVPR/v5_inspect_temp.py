import getpass,paramiko,shlex
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False);p=None
cmd="cd ~/SVR-Depth/velodepth && sed -n '295,365p' experiments/p1/gt4a2_c3_closed_loop.py && sed -n '240,455p' experiments/p1/gt4a2_y04_v4_bonn_train_v0_2.py"
_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd));print(o.read().decode(errors='replace'));print(e.read().decode(errors='replace'));c.close()
