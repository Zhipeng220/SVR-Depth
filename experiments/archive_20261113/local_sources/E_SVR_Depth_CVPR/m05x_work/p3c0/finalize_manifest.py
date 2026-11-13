import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'docs/m05x/p3c0';p3b=ROOT/'docs/m05x/p3b';m=json.loads((p3b/'evidence_manifest.json').read_text(encoding='utf8'));a=json.loads((out/'feature_direction_audit.json').read_text(encoding='utf8'))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
inputs=['docs/m05x/p2b/P2B_REPORT.md','docs/m05x/p2b/evidence_manifest.json','docs/m05x/p2b/sequence_manifest.json','docs/m05x/p3a/P3A_REPORT.md','docs/m05x/p3a/independent_candidate_inventory.csv','docs/m05x/p3b/P3B_REPORT.md','docs/m05x/p3b/evidence_manifest.json','docs/m05x/p3b/dose_response_sequence.csv','docs/m05x/p3b/source_reference/decoder_velo.py','docs/m05x/p3b/source_reference/persistent_feature_repair.py']
outputs=['AUDIT_SCOPE.md','P3C0_REPORT.md','feature_direction_audit.json','candidate_direction_inventory.csv','candidate_cpu_geometry.json','candidate_cpu_geometry.log','head_parameter_audit.json','head_parameter_audit.log','dose_response_shape_analysis.csv','dose_shape_summary.json','independent_data_audit.csv','independent_data_summary.json','video_metadata_remote.json','video_metadata_remote.log','video_hash_audit.json','video_hash_audit.log']
scripts=['audit_features.py','analyze_curves.py','audit_video_remote.py','audit_hash_remote.py','audit_heads_remote.py','audit_candidate_cpu_remote.py','build_data_audit.py','finalize_feature_audit.py','test_audit.py','finalize_manifest.py']
anchors={}
for x in a['anchors']:
 aid=x['anchor_id'];rec=m['anchors'][aid];anchors[aid]={'npz_sha256':rec['files']['evidence_arrays.npz'],'metadata_sha256':rec['files']['capture_metadata.json'],'feature_arrays':{k:v for k,v in rec['arrays'].items() if k.startswith(('committed_O__feature','committed_T__feature','committed_A0__feature','committed_A1__feature','committed_A2__feature','committed_A3__feature','committed_A4__feature'))}}
manifest={'scope':'P3-C0 CPU/read-only; no GPU, training, state commit or direction intervention','input_files_sha256':{p:sha(ROOT/p) for p in inputs},'output_files_sha256':{p:sha(out/p) for p in outputs},'script_sha256':{p:sha(ROOT/'m05x_work/p3c0'/p) for p in scripts},'p3b_anchor_count':len(anchors),'anchors':anchors,'cpu_candidate_audit':'three fixed pilot anchors; no GT, state commit or future replay','data_hash_audit':'all encoded RGB/depth in same-name train/val pairs only; no decoding or cross-video global comparison','independent_validation_certified':0,'independent_final_test_certified':0}
(out/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('EVIDENCE_MANIFEST_PASS',len(anchors),'anchors',len(outputs),'outputs')
