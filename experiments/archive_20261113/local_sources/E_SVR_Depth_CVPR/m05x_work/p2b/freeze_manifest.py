import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'docs/m05x/p2b'
scan=json.loads((out/'scan_raw.json').read_text(encoding='utf-8'))
targets={'Bonn':5,'TUM':8,'ScanNet':8};p2a={'Bonn':'moving_obstructing_box','TUM':'/scratch/lpiccinelli/datasets/tum/outputs/fr2_360_hemisphere_val','ScanNet':'scene0707_00_val'}
m04={'Bonn':{'moving_obstructing_box','person_tracking2'},'TUM':{'/scratch/lpiccinelli/datasets/tum/outputs/fr2_360_hemisphere_val','/scratch/lpiccinelli/datasets/tum/outputs/fr2_coke_val'},'ScanNet':{'scene0707_00_val','scene0708_00_val'}}
sequences={};exclusions={};anchors=[]
for dataset,n in targets.items():
 items=scan['datasets'][dataset];selected=[x for x in items if x['eligible']][:n];exclusions[dataset]=[x for x in items if not x['eligible']]
 if len(selected)<n:raise ValueError(f'{dataset} insufficient eligible sequences: {len(selected)}/{n}')
 sequences[dataset]=[]
 for i,item in enumerate(selected):
  seq=item['sequence'];entry={**item,'pilot_p2a_exposed':seq==p2a[dataset],'m04_exposed':seq in m04[dataset],'sequence_index':i}
  sequences[dataset].append(entry)
  for age in (1,4):anchors.append({'anchor_id':f'{dataset}_{i:02d}_age{age}','dataset':dataset,'sequence':seq,'sequence_index':i,'cycle':0,'age':age,'anchor_index':age,'horizons':[1,2,3,4],'K':16,'pilot_p2a_exposed':entry['pilot_p2a_exposed'],'m04_exposed':entry['m04_exposed']})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'stage':'P2-B','locked_before_ot_inference':True,'selection_rule_sha256':sha(out/'SELECTION_RULE.md'),'scan_sha256':sha(out/'scan_raw.json'),'sequences':sequences,'excluded_candidates':exclusions,'candidate_counts':{d:scan['datasets'].get(d+'_candidate_count',5) for d in targets},'target_counts':targets,'checkpoint_sha256':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251','backbone_sha256':'615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e','ages':[1,4],'K':16,'horizons':[1,2,3,4],'bootstrap':{'resampling_unit':'sequence','resamples':10000,'seed':20261025,'ci':'percentile 2.5%, 97.5%'},'data_role':'development; no claim of untouched blind test','historical_m05_formal':'BLOCKED'}
for p,obj in [(out/'sequence_manifest.json',manifest),(out/'anchor_manifest.json',{'anchors':anchors})]:
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
print('FROZEN', {k:len(v) for k,v in sequences.items()}, 'anchors',len(anchors))
