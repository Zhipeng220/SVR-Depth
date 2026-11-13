"""CPU-only parent; launches strategy processes sequentially."""
import subprocess,sys,os,json,pathlib,time
root=pathlib.Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False);worker=pathlib.Path(__file__).with_name('p5norm_v2_worker.py');events=[]
for mode in ('A','C'):
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';log=root/(mode+'.log')
 with log.open('w') as f:
  p=subprocess.Popen([sys.executable,str(worker),mode,str(root/mode)],stdout=f,stderr=subprocess.STDOUT,env=env);rc=p.wait()
 events.append({'strategy':mode,'pid':p.pid,'returncode':rc,'completed_at':time.time()});(root/'processes.json').write_text(json.dumps(events,indent=2))
 if rc:raise SystemExit('STOP: worker '+mode+' failed')
 stat=json.loads((root/mode/'manifest.json').read_text())
 if stat.get('status')!='PASS':raise SystemExit('STOP: invalid worker result')
 subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],stdout=(root/(mode+'_after_gpu.txt')).open('w'),check=True)
print('A/C completed; validate cross-process branch inputs independently')
