import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
results={};indexes={}
for age in (1,4):
    d=root/f'verification_age{age}'
    results[str(age)]=json.loads((d/'invariant_validation.json').read_text(encoding='utf-8'))
    indexes[str(age)]=json.loads((d/'evidence_manifest.json').read_text(encoding='utf-8'))
status='PASS' if all(r['status']=='PASS' for r in results.values()) else 'BLOCKED'
summary={'status':status,'scope':'P1.5 new run; two anchors age 1 and age 4 in one Bonn development sequence; no inference about historical run','per_age':results,'historical_formal_status':'BLOCKED: historical pixel masks, D1 tensors, and exact generator hash not recovered'}
(root/'invariant_validation.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
manifest={'scope':'P1.5 new runs, per-array shape/dtype/SHA256 and raw NPZ file hashes','per_age':indexes,'verifier_sha256':hashlib.sha256((root/'verify_evidence.py').read_bytes()).hexdigest()}
(root/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(status)
