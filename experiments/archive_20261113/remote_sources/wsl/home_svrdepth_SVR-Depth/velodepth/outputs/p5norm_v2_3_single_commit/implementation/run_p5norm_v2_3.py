import os,sys,csv,json,subprocess
from pathlib import Path
from p5norm_v2_3_utility_eval import evaluate
base=Path('/home/svrdepth/SVR-Depth/velodepth/outputs');root=base/'p5norm_v2_3_single_commit';root.mkdir(exist_ok=False);anchors=list(csv.DictReader((base/'p5norm_v2_ac_pilot/sequence_selection/pilot_anchor4_manifest.csv').open()));assert len(anchors)==18
smoke=[]
for dom in ['Bonn','TUM','ScanNet']:smoke.append(next(x for x in anchors if x['dataset']==dom))
ordered=smoke+[x for x in anchors if x not in smoke];records=[];worker=Path(__file__).with_name('p5norm_v2_3_single_commit_worker.py')
for i,a in enumerate(ordered):
 folder=root/a['dataset']/Path(a['sequence_id']).name/('anchor_'+a['anchor_idx']);folder.parent.mkdir(parents=True,exist_ok=True);log=folder.parent/('anchor_'+a['anchor_idx']+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
 with log.open('x') as f:p=subprocess.Popen([sys.executable,str(worker),a['dataset'],a['sequence_id'],a['anchor_idx'],str(folder)],env=env,stdout=f,stderr=subprocess.STDOUT);rc=p.wait()
 if rc:raise SystemExit('STOP: '+str(log))
 r=evaluate(a,folder);records.append(r);(root/'partial_results.json').write_text(json.dumps(records,indent=2,allow_nan=False));print('COMPLETE',i+1,a['dataset'],a['anchor_id'],'utility',r['utility_absrel'],flush=True)
 if i==2:print('THREE DOMAIN SMOKES PASSED',flush=True)
print('ALL 18 COMPLETE',flush=True)
