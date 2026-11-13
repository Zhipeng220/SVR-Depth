from pathlib import Path
import csv,torch,numpy as np,json
base=Path('/home/svrdepth/SVR-Depth/velodepth/outputs');a=base/'p5norm_eng02_fixed_policy_rollout';b=base/'p5norm_eng04_layer_marginal_ablation';out=base/'p5norm_eng05_factorial_screening';out.mkdir(exist_ok=False)
def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def check(x,msg):
 if not x:raise ValueError(msg)
try:
 states=[torch.load(p/'initial_parent.pt',map_location='cpu',weights_only=True) for p in [a,b]]
 check(all(torch.equal(states[0][k],states[1][k]) for k in ['radius','confidence','rays_map']) and all(torch.equal(x,y) for x,y in zip(states[0]['features'],states[1]['features'])),'initial state differs')
 preds=[np.load(p/'first_prediction.npy') for p in [a,b]];check(np.array_equal(*preds),'first prediction differs')
 entries=[(a,'HOLD'),(a,'LAYER1_ONLY')]+[(b,x) for x in ['FULL_COMMIT','NO_L1','NO_L2','NO_L3','NO_L4']];reference=None;details=[]
 for folder,name in entries:
  m=read(folder/name/'metrics.csv');t=read(folder/name/'trace.csv');check(len(m)==8 and len(t)==32,'incomplete '+name);frames=[(x['frame'],x['valid_pixels']) for x in m];times=[x['timestamp'] for x in t if x['layer']=='1'];check(len(times)==8,'missing timestamps '+name)
  if reference is None:reference=(frames,times)
  check((frames,times)==reference,'frame/mask count/timestamps differ '+name);details.append({'policy':name,'frames':len(m),'timestamps':times})
 src=[(folder/name).read_text() for folder,name in [(a,'eng02_rollout.py'),(b,'eng04_rollout.py')]]
 required=['y04_v5_preserve_seed20261025_20261109_112607/final.pt','y04_v4_manifest_candidate_20261108_190940/person_tracking2_pairs_v4_candidate.csv','p5norm_v2_4_independent_replication/runs/Bonn/person_tracking2/hold/anchor_1_tensors.pt','(480,640)','(434,574)','m05.make_prediction(model,gt2,native,shape)','gt2.calculate_metrics(gt,pred)','np.isfinite(pred).all() and (pred>0).all()']
 for x in required:check(all(x in z for z in src),'source config mismatch '+x)
 # Positive, finite predictions make calculate_metrics valid mask exactly GT>0 & finite GT.
 g0=read(a/'FULL_COMMIT'/'metrics.csv');g1=read(b/'FULL_COMMIT'/'metrics.csv');check(all(x['frame']==y['frame'] and x['valid_pixels']==y['valid_pixels'] and abs(float(x['AbsRel'])-float(y['AbsRel']))<1e-12 and abs(float(x['RMSE'])-float(y['RMSE']))<1e-12 for x,y in zip(g0,g1)),'FULL baseline metrics mismatch')
 result={'result':'PASS','parent_exact':True,'first_prediction_exact':True,'metrics_full_reproduced':True,'frames':reference[0],'timestamps':reference[1],'valid_mask_evidence':'same manifest GT; calculate_metrics GT>0 & finite; scripts require every prediction finite and >0; therefore identical mask; counts equal','source_checkpoint_identity':'same checkpoint path and frozen base snapshot; historical bytes not independently compared','policies':details}
except Exception as e:
 result={'result':'NO-GO','reason':str(e)}
(out/'P5NORM_ENG05_BASELINE_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if result['result']!='PASS':raise SystemExit(1)
