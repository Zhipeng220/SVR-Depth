import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'docs/m05x/p3b';path=out/'evidence_manifest.json';m=json.loads(path.read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
files=['P3B_REPORT.md','protocol_lock.json','SEMANTICS_AUDIT.md','state_semantics_ranges.json','dose_response_anchor.csv','dose_response_sequence.csv','dose_response_dataset.csv','alpha_endpoint_validation.json','invariant_validation.json','dose_response_curves.png','dose_response_curves.pdf','execution_progress.json']
m['p3b_output_sha256']={name:sha(out/name) for name in files};m['source_sha256']={name:sha(root/'m05x_work/p3b'/name) for name in ['capture_p3b.py','verify_p3b.py','stats_p3b.py','test_verify_p3b.py','test_stats_p3b.py','run_p3b.py','state_semantics_audit.py','finalize_manifest.py']};m['p2b_anchor_manifest_sha256']=sha(root/'docs/m05x/p2b/anchor_manifest.json');m['p2b_evidence_manifest_sha256']=sha(root/'docs/m05x/p2b/evidence_manifest.json');path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('manifest finalized',len(m['anchors']),'anchors')
