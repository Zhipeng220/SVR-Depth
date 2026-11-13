import json,subprocess,sys,time,traceback
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'docs/m05x/p3c1_formal';progress=out/'execution_progress.json';log=root/'m05x_work/p3c1_formal/auto_finalize.log'
with log.open('a',encoding='utf8',buffering=1) as f:
 f.write('Automatic finalization started; waits for runner completion or recorded early stop.\n')
 while True:
  try:
   d=json.loads(progress.read_text(encoding='utf8'))
   if d.get('stopped_early') or len(d.get('completed',[]))>=39:
    r=subprocess.run([sys.executable,str(root/'m05x_work/p3c1_formal/summarize_formal.py')],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
    f.write(r.stdout+'\nExit code: '+str(r.returncode)+'\n');break
  except (OSError,ValueError):f.write('Progress temporarily unavailable; retrying.\n')
  time.sleep(30)
