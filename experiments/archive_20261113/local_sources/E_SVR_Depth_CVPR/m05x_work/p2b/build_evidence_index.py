import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'docs/m05x/p2b';anchors=json.loads((out/'anchor_manifest.json').read_text(encoding='utf-8'))['anchors'];checks={};files={};missing=[]
for a in anchors:
 name=a['anchor_id'];directory=out/'runs'/name;v=directory/'verification/invariant_validation.json';m=directory/'verification/evidence_manifest.json'
 if not v.is_file() or not m.is_file():missing.append(name);continue
 checks[name]=json.loads(v.read_text(encoding='utf-8'));files[name]=json.loads(m.read_text(encoding='utf-8'))
fail=[k for k,v in checks.items() if v.get('status')!='PASS']
status='PASS' if not missing and not fail and len(checks)==len(anchors) else 'BLOCKED'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'invariant_validation.json').write_text(json.dumps({'status':status,'expected_anchors':len(anchors),'verified':len(checks),'missing':missing,'invalid':fail,'anchors':checks},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(out/'evidence_manifest.json').write_text(json.dumps({'status':status,'sequence_manifest_sha256':sha(out/'sequence_manifest.json'),'anchor_manifest_sha256':sha(out/'anchor_manifest.json'),'capture_sha256':sha(root/'m05x_work/p2b/capture_p2b.py'),'verifier_sha256':sha(root/'m05x_work/p2b/verify_p2b.py'),'deduplication':'Logical array aliases refer to byte-identical dtype/shape/content arrays; original states and all logical keys can be reconstructed losslessly from NPZ plus capture_metadata.json.','anchors':files},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(status,'verified',len(checks),'missing',len(missing),'invalid',len(fail))
