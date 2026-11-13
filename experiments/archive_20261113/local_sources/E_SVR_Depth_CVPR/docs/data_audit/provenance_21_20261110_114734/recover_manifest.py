import csv,hashlib,json
from collections import Counter
from pathlib import Path
root=Path.cwd();out=root/'docs/data_audit/provenance_21_20261110_114734';out.mkdir(parents=True,exist_ok=True)
files=['docs/m05x/p2b/P2B_REPORT.md','docs/m05x/p2b/sequence_manifest.json','docs/m05x/p2b/anchor_manifest.json','docs/m05x/p2b/sequence_level_results.csv','docs/m05x/p2b/paired_anchor_results.csv','docs/m05x/p2b/dataset_summary.json','docs/m05x/p2b/evidence_manifest.json','docs/m05x/p3b/P3B_REPORT.md','docs/m05x/p3b/dose_response_sequence.csv','docs/m05x/p3b/dose_response_anchor.csv','docs/m05x/p3b/evidence_manifest.json','docs/m05x/p3a/P3A_REPORT.md','docs/m05x/p3c0/P3C0_REPORT.md']
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
manifest=json.loads((root/files[1]).read_text(encoding='utf8'));anchors=json.loads((root/files[2]).read_text(encoding='utf8'))['anchors'];sequences=[{'dataset':ds,'sequence':x['sequence'],'sequence_index':x['sequence_index']} for ds,items in manifest['sequences'].items() for x in items]
def key(r):return (r['dataset'],r['sequence'])
if len(sequences)!=21 or len({key(x) for x in sequences})!=21 or Counter(x['dataset'] for x in sequences)!={'Bonn':5,'TUM':8,'ScanNet':8}:raise RuntimeError('selected sequence manifest not 5/8/8 unique')
for path in [root/files[3],root/files[8],root/'docs/m05x/p3a/sequence_feature_effect.csv']:
 actual={key(x) for x in readcsv(path)}
 if actual!={key(x) for x in sequences}:raise RuntimeError('sequence membership mismatch '+str(path))
if len(anchors)!=42 or len({x['anchor_id'] for x in anchors})!=42:raise RuntimeError('anchor count or uniqueness mismatch')
if {key(x) for x in anchors}!={key(x) for x in sequences} or any(sum(key(a)==key(x) for a in anchors)!=2 for x in sequences):raise RuntimeError('anchor membership mismatch')
for path in [root/files[4],root/files[9]]:
 actual={r['anchor_id'] for r in readcsv(path)}
 if actual!={r['anchor_id'] for r in anchors}:raise RuntimeError('anchor membership mismatch '+str(path))
info={'status':'PASS','sequences':sequences,'sequence_counts':dict(Counter(x['dataset'] for x in sequences)),'anchor_count':len(anchors),'sequence_manifest_sha256':hashlib.sha256((root/files[1]).read_bytes()).hexdigest(),'files':[{'path':f,'exists':(root/f).is_file(),'sha256':hashlib.sha256((root/f).read_bytes()).hexdigest() if (root/f).is_file() else None} for f in files]}
(out/'selected_sequences.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('21 SEQUENCE MANIFEST',info['status'],info['sequence_counts'],'anchors',len(anchors))

