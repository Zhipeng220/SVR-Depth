import csv,json,io,math,re,statistics
from pathlib import Path
import numpy as np,h5py,torch,cv2
from PIL import Image
from scipy.stats import norm
base=Path('/home/svrdepth/SVR-Depth/velodepth/outputs');data=Path('/home/svrdepth/datasets');out=base/'p5norm_v2_6_preregistration';out.mkdir(exist_ok=False)
prior=list(csv.DictReader((base/'p5norm_v2_5_mechanism_consolidation/P5NORM_V2_5_SEQUENCE_LEVEL_COMBINED.csv').open()));p3={d:set() for d in ['Bonn','TUM','ScanNet','Sintel']}
for r in prior:
 s=Path(r['sequence_id']).name.removesuffix('_val').removesuffix('_train');p3[r['dataset']].add(s)
rows=[]
for d in ['ScanNet','TUM','Sintel']:
 with h5py.File(data/'velodepth_evaluation'/(d+'.hdf5'),'r') as h:
  md=json.loads(bytes(h['sequences_.json' if d=='ScanNet' else 'sequences.json'][:]).decode())
  for seq,v in md.items():
   short=Path(seq).name.removesuffix('_train').removesuffix('_val');physical=re.sub(r'_\d\d$','',short) if d=='ScanNet' else short
   exposed=(short in p3[d]) or (d=='ScanNet' and any(re.sub(r'_\d\d$','',x)==physical for x in p3[d]))
   z=dict(dataset=d,domain=d,scene_id=short,sequence_id=seq,physical_scene_id=physical,P3B_exposure='YES' if exposed else 'NOT_FOUND_IN_21_SEQUENCE_LIST',P5_exposure='YES' if exposed else 'UNKNOWN',frame_count=len(v.get('image',[])),RGB_valid='UNKNOWN',GT_valid='UNKNOWN',intrinsics_valid='UNKNOWN',overlap_status='EXPOSED' if exposed else 'UNKNOWN_PHYSICAL_AND_PROJECT_HISTORY',eligible='NO',exclusion_reason='',time_order='UNKNOWN',loader_status='UNKNOWN',intrinsics_source='')
   try:
    images=v['image'];depth=v['depth'];assert len(images)==len(depth) and len(images)>=9,'insufficient/mismatched frames';pairs=[]
    for i in [1,4,5,8]:
     rgb=np.array(Image.open(io.BytesIO(h[images[i]][:].tobytes())).convert('RGB'));dep=np.array(Image.open(io.BytesIO(h[depth[i]][:].tobytes())))
     if dep.ndim==3:dep=dep[:,:,0].astype(np.int32)|(dep[:,:,1].astype(np.int32)<<8)|(dep[:,:,2].astype(np.int32)<<16)
     assert dep.shape==rgb.shape[:2],'RGB GT shape mismatch';
     if d in ['ScanNet','TUM']:
      g=dep.astype(np.float32)/1000;mask=np.isfinite(g)&(g>(.01 if d=='ScanNet' else .001))&(g<10);assert mask.any(),'no valid GT'
     else:assert np.isfinite(dep).any() and (dep>0).any(),'no positive raw GT'
     pairs.append((images[i],depth[i]))
    z['RGB_valid']='SAMPLED_4_PASS';z['GT_valid']='SAMPLED_4_PASS_EXISTING_RULE' if d!='Sintel' else 'SAMPLED_RAW_ONLY_SCALE_RULE_UNKNOWN'
    if 'K' in v:
     k=np.array([v['K'][i] for i in [1,4,5,8]],float);assert k.shape==(4,3,3) and np.isfinite(k).all() and (k[:,0,0]>0).all() and (k[:,1,1]>0).all();z['intrinsics_valid']='SAMPLED_4_PASS';z['intrinsics_source']='K'
    elif 'camera_params' in v:
     k=np.array([v['camera_params'][i] for i in [1,4,5,8]],float);assert np.isfinite(k).all() and (k[:,0]>0).all() and (k[:,1]>0).all();z['intrinsics_valid']='SAMPLED_4_PASS';z['intrinsics_source']='camera_params'
    ids=[Path(images[i]).stem for i in [1,4,5,8]];z['time_order']='METADATA_ORDER_ONLY_UNVERIFIED';z['loader_status']='EXISTING_V2_HDF5_PATH' if d in ['ScanNet','TUM'] else 'V2_HDF5_DOMAIN_UNSUPPORTED'
   except Exception as e:z['exclusion_reason']='DATA_CHECK_FAILED: '+str(e)
   if exposed:z['exclusion_reason']='P3-B/P5 exposed physical scene'
   elif not z['exclusion_reason']:z['exclusion_reason']='P5 history, physical independence and timestamp order unverified'+('; Sintel evaluation/loader unsupported' if d=='Sintel' else '')
   rows.append(z)
for p in sorted((data/'bonn').glob('rgbd_bonn_*')):
 if not p.is_dir():continue
 short=p.name.removeprefix('rgbd_bonn_');exposed=short in p3['Bonn'];z=dict(dataset='Bonn',domain='Bonn',scene_id=short,sequence_id=short,physical_scene_id='UNKNOWN',P3B_exposure='YES' if exposed else 'NOT_FOUND_IN_21_SEQUENCE_LIST',P5_exposure='YES' if exposed else 'UNKNOWN',frame_count=len(list((p/'rgb').glob('*.png'))),RGB_valid='UNKNOWN',GT_valid='UNKNOWN',intrinsics_valid='UNKNOWN',overlap_status='EXPOSED' if exposed else 'UNKNOWN_PHYSICAL_AND_PROJECT_HISTORY',eligible='NO',exclusion_reason='P3-B/P5 exposed' if exposed else 'physical scene, source history, calibration and loader manifest not established',time_order='UNKNOWN',loader_status='UNKNOWN',intrinsics_source='');rows.append(z)
fields=['dataset','domain','scene_id','sequence_id','physical_scene_id','P3B_exposure','P5_exposure','frame_count','RGB_valid','GT_valid','intrinsics_valid','overlap_status','eligible','exclusion_reason','time_order','loader_status','intrinsics_source']
for name,rr in [('P5NORM_V2_6_DATA_ELIGIBILITY.csv',rows),('P5NORM_V2_6_FROZEN_SCENE_LIST.csv',[r for r in rows if r['eligible']=='YES'])]:
 with (out/name).open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
counts={d:dict(sequence_records=sum(r['dataset']==d for r in rows),unexposed_proxies=len({r['physical_scene_id'] for r in rows if r['dataset']==d and r['P3B_exposure']!='YES' and r['physical_scene_id']!='UNKNOWN'}),sample4_rgb=sum(r['dataset']==d and r['RGB_valid']=='SAMPLED_4_PASS' for r in rows),sample4_gt=sum(r['dataset']==d and r['GT_valid']=='SAMPLED_4_PASS_EXISTING_RULE' for r in rows),intrinsics=sum(r['dataset']==d and r['intrinsics_valid']=='SAMPLED_4_PASS' for r in rows),exposed=sum(r['dataset']==d and r['P3B_exposure']=='YES' for r in rows)) for d in ['Bonn','TUM','ScanNet','Sintel']}
print('INVENTORY',json.dumps(counts));
stats=[]
for p in (base/'p5norm_v2_4_independent_replication/runs').glob('*/*/hold/anchor_*_tensors.pt'):
 t=torch.load(p,map_location='cpu',weights_only=True);r=t['radius'].float();c=t['confidence'].float();e=math.exp(2);diff=(r-e*c).abs();rel=diff/(e*c).abs().clamp_min(1e-20);stats.append(dict(path=str(p),n=r.numel(),shape=list(r.shape),dtype=str(t['radius'].dtype),min_conf=float(c.min()),max_conf=float(c.max()),mean_ratio=float(r.mean()/c.mean()),max_abs=float(diff.max()),mean_rel=float(rel.mean()),max_rel=float(rel.max()),pct_rel_below_001=float((rel<.001).float().mean()),pct_rel_below_005=float((rel<.005).float().mean()),count_conf_gt_e8=int((c>math.exp(8)).sum())))
(out/'radius_confidence_observations.json').write_text(json.dumps(stats,indent=2),encoding='utf8');print('TRACE',json.dumps({'files':len(stats),'elements':sum(z['n'] for z in stats),'max_rel':max(z['max_rel'] for z in stats),'max_abs':max(z['max_abs'] for z in stats),'mean_ratio_range':[min(z['mean_ratio'] for z in stats),max(z['mean_ratio'] for z in stats)],'conf_gt_e8':sum(z['count_conf_gt_e8'] for z in stats),'dtypes':list({z['dtype'] for z in stats})}))
def fisher_power(n,rho,k=0):
 se=1/math.sqrt(n-k-3);m=np.arctanh(rho)/se;return float(norm.sf(1.95996398454-m)+norm.cdf(-1.95996398454-m))
print('POWER',json.dumps({str(n):{'h1_r04':fisher_power(n,.4),'h1_r03':fisher_power(n,.3),'h2_r03_cov3':fisher_power(n,.3,3)} for n in [30,40,47,50,60,85,90,100,120]}))
