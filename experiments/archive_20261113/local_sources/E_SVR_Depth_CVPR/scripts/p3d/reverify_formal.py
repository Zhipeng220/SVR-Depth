import json,subprocess,sys
from pathlib import Path
root=Path.cwd();out=root/'docs/p3d';out.mkdir(parents=True,exist_ok=True);formal=root/'docs/m05x/p3c1_formal';anchors=json.loads((formal/'anchor_manifest.json').read_text(encoding='utf8'))['formal_anchors'];results=[]
for a in anchors:
 aid=a['anchor_id'];source=formal/'runs'/aid;target=out/'reverification'/aid;target.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,str(root/'m05x_work/p3c1_formal/verify_formal.py'),str(source),'--output-dir',str(target)]
 r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf8',errors='replace');(target/'verification.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf8')
 result={'anchor_id':aid,'exit_code':r.returncode,'status':'PASS' if r.returncode==0 else 'BLOCKED'};results.append(result);print(len(results),aid,result['status'],flush=True)
 (out/'formal_reverification_progress.json').write_text(json.dumps({'results':results,'expected':len(anchors)},ensure_ascii=False,indent=2),encoding='utf8')
 if r.returncode:break
print('REVERIFICATION',sum(x['status']=='PASS' for x in results),'/',len(anchors),flush=True)
