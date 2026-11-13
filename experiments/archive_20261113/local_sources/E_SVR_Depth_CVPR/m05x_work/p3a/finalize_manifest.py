import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'docs/m05x/p3a';path=out/'evidence_manifest.json';m=json.loads(path.read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
outputs=['P3A_REPORT.md','sequence_feature_effect.csv','correlation_summary.json','leave_one_sequence_out.csv','feature_norm_distribution.csv','anchor_feature_audit.json','feature_update_scatter.png','feature_update_scatter.pdf','candidate_metadata_raw.json','candidate_metadata_error.log','independent_candidate_inventory.csv','independent_candidate_summary.json']
m['p3a_outputs_sha256']={name:sha(out/name) for name in outputs};m['support_scripts_sha256']={name:sha(root/'m05x_work/p3a'/name) for name in ['analyze.py','test_analyze.py','candidate_inventory.py','candidate_metadata_remote.py','finalize_manifest.py']};m['candidate_usage_source_archive_sha256']=sha(root/'SVR-Depth_实验证据快照_20261109.zip');path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('manifest finalized',len(m['anchor_sources']),'anchors')
