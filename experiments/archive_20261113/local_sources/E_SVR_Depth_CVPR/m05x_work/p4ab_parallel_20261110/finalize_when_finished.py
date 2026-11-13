import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];work=root/'m05x_work/p4ab_parallel_20261110';progress=work/'capture_progress.json';log=work/'automatic_finalization.log'
with log.open('a',encoding='utf8',buffering=1) as f:
 f.write('Waiting for 39 verified captures or recorded error; no model training unless certified groups exist.\n')
 while True:
  try:
   p=json.loads(progress.read_text(encoding='utf8'));records=p.get('records',[]);error=p.get('error','')
   if error or len(records)>=39:
    f.write('Capture finished or stopped: '+str(len(records))+' error='+str(error)+'\n')
    for script in ['build_dataset.py','reports.py']:
     r=subprocess.run([sys.executable,str(work/script)],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
     f.write(script+' exit='+str(r.returncode)+'\n'+r.stdout+'\n')
     if r.returncode:break
    break
  except (OSError,ValueError) as e:f.write('Waiting for readable progress: '+repr(e)+'\n')
  time.sleep(30)

