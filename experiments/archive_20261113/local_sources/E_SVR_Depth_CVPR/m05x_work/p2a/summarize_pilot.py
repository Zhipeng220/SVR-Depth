import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'docs/m05x/p2a';execution=json.loads((out/'pilot_execution.json').read_text(encoding='utf-8'))
records=[];validations={};manifests={};invalid=[]
for dataset in ('Bonn','TUM','ScanNet'):
 for age in (1,4):
  key=f'{dataset}_age{age}';d=out/'runs'/key;v=d/'verification/invariant_validation.json';m=d/'verification/evidence_manifest.json';row={'dataset':dataset,'age':age,'status':'BLOCKED','reason':'未执行或未通过独立检查'}
  if v.exists():
   obj=json.loads(v.read_text(encoding='utf-8'));validations[key]=obj;row['status']=obj['status']
   if obj['status']=='PASS':
    meta=json.loads((d/'capture_metadata.json').read_text(encoding='utf-8'));row.update(sequence=meta['sequence'],reason='',**obj['metrics_absrel'],valid_pixels=json.dumps(obj['valid_pixels'],ensure_ascii=False),feature_update_norms=json.dumps(obj['details']['feature_update_norms'],ensure_ascii=False),identity=json.dumps(meta['identity'],ensure_ascii=False))
   else: row['reason']='独立检查未通过或证据缺失'
  if m.exists():manifests[key]=json.loads(m.read_text(encoding='utf-8'))
  if row['status']!='PASS':invalid.append({'dataset':dataset,'age':age,'status':row['status'],'reason':row['reason']})
  records.append(row)
columns=['dataset','sequence','age','status','reason','O_H1','T_H1','T_minus_O_H1','O_H4','T_H4','T_minus_O_H4','valid_pixels','feature_update_norms','identity']
with (out/'pilot_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,columns,extrasaction='ignore');w.writeheader();w.writerows(records)
status='PASS' if len(validations)==6 and not invalid else 'BLOCKED'
(out/'invariant_validation.json').write_text(json.dumps({'stage':'P2-A','status':status,'anchors':validations},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(out/'invalid_anchor_report.json').write_text(json.dumps({'status':status,'invalid_or_blocked':invalid,'execution':execution},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(out/'evidence_manifest.json').write_text(json.dumps({'anchors':manifests,'pilot_protocol_sha256':hashlib.sha256((out/'pilot_protocol.json').read_bytes()).hexdigest(),'capture_sha256':hashlib.sha256((root/'m05x_work/p2a/capture_p2a.py').read_bytes()).hexdigest(),'verifier_sha256':hashlib.sha256((root/'m05x_work/p2a/verify_p2a.py').read_bytes()).hexdigest()},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(status,'valid',sum(x['status']=='PASS' for x in records),'invalid_or_blocked',len(invalid))
