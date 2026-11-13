"""Create matched V5 dataset only after each independent feature-capture check."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
root=Path.cwd();w=root/'m05x_work/p4ab_parallel_20261110';out=root/'docs/p4ab';out.mkdir(parents=True,exist_ok=True);formal=root/'docs/m05x/p3c1_formal'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
selection=json.loads((w/'capture_selection.json').read_text(encoding='utf8'));progress=json.loads((w/'capture_progress.json').read_text(encoding='utf8'));lock=json.loads((w/'protocol_lock.json').read_text(encoding='utf8'))
known={r['anchor_id']:r for r in progress['records']};pairs={r['anchor_id']:r for r in readcsv(formal/'paired_anchor_results.csv')};geometry={r['anchor_id']:r for r in readcsv(formal/'global_vs_layerwise_geometry.csv')};rows=[];invalid=[]
group_file=w/'certified_physical_groups.json';groups=json.loads(group_file.read_text(encoding='utf8')) if group_file.exists() else {}
for item in selection['records']:
 aid=item['anchor_id'];r=known.get(aid)
 try:
  if not r or r['status']!='PASS':raise ValueError('C3 capture missing or invalid')
  p=Path(r['path']);verify=(w/'reverified'/aid/'independent_validation.json') if r['source']=='reused' else (p/'audit/independent_validation.json')
  v=json.loads(verify.read_text(encoding='utf8'))
  if v['status']!='PASS':raise ValueError('independent verification not PASS')
  result=json.loads((p/'metrics/result.json').read_text(encoding='utf8'));zpath=p/'capture/capture.npz'
  if sha(zpath)!=result['capture_npz_sha256'] or not result['pass']:raise ValueError('capture digest/status mismatch')
  if result['anchor']!=aid or result['identity']['checkpoint']!=lock['checkpoint_sha256']:raise ValueError('anchor/model identity mismatch')
  lab=pairs[aid];geom=geometry[aid];old=json.loads((formal/'runs'/aid/'verification/invariant_validation.json').read_text(encoding='utf8'))
  if old['status']!='PASS' or not old['checks']['gt_and_metric_masks_equal'] or not old['checks']['prior_endpoints_exact']:raise ValueError('historical mask/protocol invalid')
  o=float(lab['O_H4_AbsRel']);commit=float(lab['V5_H4_AbsRel']);utility=o-commit
  if not all(math.isfinite(x) for x in [o,commit,utility]) or int(lab['O_H4_valid_pixels'])!=int(lab['V5_H4_valid_pixels']):raise ValueError('invalid/misaligned H4 mask')
  norms=[]
  with np.load(zpath,allow_pickle=False) as z:
   for i in range(4):
    q=z['delta_'+str(i)].astype(np.float64);n=float(np.sqrt(np.sum(q*q)))
    if not math.isfinite(n) or not math.isclose(n,float(geom[f'layer{i}_v5_native_l2']),rel_tol=1e-8,abs_tol=1e-8):raise ValueError('norm mismatch layer '+str(i))
    norms.append(n)
  glob=math.sqrt(sum(x*x for x in norms))
  if not math.isclose(glob,float(geom['v5_native_l2']),rel_tol=1e-8,abs_tol=1e-8):raise ValueError('global norm mismatch')
  g=groups.get(item['dataset']+'::'+item['sequence'],{});cert=g.get('status')=='CERTIFIED' and bool(g.get('physical_group_id')) and bool(g.get('provenance'))
  row={'anchor_id':aid,'dataset':item['dataset'],'sequence_id':item['sequence'],'age':item['age'],'physical_group_id':g.get('physical_group_id','') if cert else '', 'physical_group_status':'CERTIFIED' if cert else 'UNKNOWN','candidate':'V5-native','matched_evidence':'PASS','global_l2':glob,**{f'l{i}_l2':norms[i] for i in range(4)},**{f'c3_l{i}':result['layers'][i]['c3'] for i in range(4)},'O_H4_AbsRel':o,'V5_H4_AbsRel':commit,'U_H4':utility,'H4_valid_pixels':int(lab['O_H4_valid_pixels']),'capture_sha256':sha(zpath),'historical_npz_sha256':item['evidence_arrays.npz'],'capture_source':r['source'],'physical_group_provenance':g.get('provenance','') if cert else ''}
  if any(row[f'c3_l{i}'] is None or not math.isfinite(row[f'c3_l{i}']) for i in range(4)):raise ValueError('undefined C3')
  rows.append(row)
 except Exception as e:invalid.append({'anchor_id':aid,'reason':str(e)})
if len({(r['anchor_id'],r['candidate']) for r in rows})!=len(rows):raise ValueError('duplicate samples')
outpath=out/'P4AB_PAIRED_DATASET.csv';fields=['anchor_id','dataset','sequence_id','age','physical_group_id','physical_group_status','candidate','matched_evidence','global_l2']+[f'l{i}_l2' for i in range(4)]+[f'c3_l{i}' for i in range(4)]+['O_H4_AbsRel','V5_H4_AbsRel','U_H4','H4_valid_pixels','capture_sha256','historical_npz_sha256','capture_source','physical_group_provenance']
with outpath.open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
state={'frozen_dataset_sha256':sha(outpath),'protocol_lock_sha256':sha(w/'protocol_lock.json'),'capture_selection_sha256':sha(w/'capture_selection.json'),'n_valid':len(rows),'n_invalid':len(invalid),'invalid':invalid,'domain_anchors':{d:sum(x['dataset']==d for x in rows) for d in ['Bonn','TUM','ScanNet']},'sequence_records':len({(r['dataset'],r['sequence_id']) for r in rows}),'physical_groups_certified':len({r['physical_group_id'] for r in rows if r['physical_group_status']=='CERTIFIED'}),'unknown_physical_rows':sum(r['physical_group_status']!='CERTIFIED' for r in rows),'complete':len(rows)==39}
(w/'paired_dataset_freeze.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('DATASET',json.dumps(state,ensure_ascii=False))

