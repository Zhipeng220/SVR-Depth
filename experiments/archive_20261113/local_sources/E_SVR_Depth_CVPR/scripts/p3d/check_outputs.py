import csv,importlib.util,json,sys
from collections import Counter
from pathlib import Path
root=Path.cwd();spec=importlib.util.spec_from_file_location('checks',root/'scripts/p3d/checks.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
base=root/'docs';reports=root/'reports/p3d_dataaudit_p4a_preparation';reports.mkdir(parents=True,exist_ok=True)
files=[base/'p3d/P3D_EVIDENCE_SYNTHESIS.md',base/'p3d/P3D_EVIDENCE_MATRIX.csv',base/'p3d/evidence_manifest.json',base/'data_audit/DATA_AUDIT_REPORT.md',base/'data_audit/DATA_SPLIT_POLICY.md',base/'p4a/P4A_PROTOCOL.md',base/'p4a/P4A_GO_NO_GO.md',base/'p4a/P4A_FEATURE_SCHEMA.csv']+[base/'data_audit'/x for x in ('video_identity_manifest.csv','temporal_overlap_audit.csv','experiment_usage_history.csv','data_split_manifest.csv','unresolved_data.csv')];c.require_files(files)
data=[readcsv(base/'data_audit'/x) for x in ('video_identity_manifest.csv','temporal_overlap_audit.csv','experiment_usage_history.csv','data_split_manifest.csv')]
for rows in data:c.unique(rows,['dataset','sequence_id'])
if any(len(x)!=280 for x in data):raise ValueError('audit row counts mismatch')
if len({frozenset((r['dataset'],r['sequence_id']) for r in rows) for rows in data})!=1:raise ValueError('audit key sets differ')
if any(r['independent_validation_certified']!='False' or r['final_test_certified']!='False' for r in data[-1]):raise ValueError('unapproved independent assignment')
matrix=readcsv(base/'p3d/P3D_EVIDENCE_MATRIX.csv');c.unique(matrix,['stage','dataset']);c.required_fields(matrix,['stage','dataset','n_sequences','n_anchors','primary','raw'])
formal=readcsv(base/'m05x/p3c1_formal/paired_anchor_results.csv');c.unique(formal,['anchor_id']);c.required_fields(formal,['anchor_id','dataset','sequence','V4L_H4_AbsRel','V5_H4_AbsRel']);assert len(formal)==39
manifest=json.loads((base/'p3d/evidence_manifest.json').read_text(encoding='utf8'))
for item in manifest['files']:
 if not item['exists']:raise FileNotFoundError(item['path'])
 c.check_sha(item['path'],item['sha256'])
remote=json.loads((base/'data_audit/remote_inventory.json').read_text(encoding='utf8'))
if remote['errors']:raise ValueError('remote scan errors')
bonn=remote['bonn'];time_unknown=[x['sequence'] for x in bonn if x.get('timestamp_audit',{}).get('rgb_monotonic_strict') is not True];depth_nondecreasing_unknown=[x['sequence'] for x in bonn if x.get('timestamp_audit',{}).get('depth_monotonic_nondecreasing') is not True]
features=readcsv(base/'p4a/P4A_FEATURE_SCHEMA.csv');c.unique(features,['feature']);c.no_leak([{'name':x['feature'],'role':'group' if x['category']=='audit_only' else 'predictor'} for x in features])
result={'status':'PASS_WITH_UNRESOLVED_DATA','required_files':len(files),'audit_rows':len(data[0]),'matrix_rows':len(matrix),'formal_anchors':len(formal),'sha_files_checked':len(manifest['files']),'bonn_rgb_nonmonotonic_or_unknown':time_unknown,'bonn_depth_nondecreasing_unconfirmed':depth_nondecreasing_unknown,'timestamp_note':'TUM/ScanNet timestamp continuity unavailable; flagged UNKNOWN; Bonn pair CSV timing does not prove physical source independence','training_started':False}
(reports/'automated_checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False))
