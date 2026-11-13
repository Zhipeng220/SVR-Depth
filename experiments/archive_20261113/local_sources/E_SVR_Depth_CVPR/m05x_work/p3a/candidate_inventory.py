"""Metadata-only candidate audit. Does not open RGB, depth, predictions or metric files."""
import csv,json,zipfile
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/m05x/p3a';meta=json.loads((OUT/'candidate_metadata_raw.json').read_text(encoding='utf8'));p2b=json.loads((ROOT/'docs/m05x/p2b/sequence_manifest.json').read_text(encoding='utf8'))
def canonical(s):
 for suffix in ('_train','_val','_test'):
  if s.endswith(suffix):return s[:-len(suffix)]
 return s
known=defaultdict(lambda:defaultdict(set))
for domain,records in p2b['sequences'].items():
 for r in records:known[domain][r['sequence']].add('P2-B')
z=zipfile.ZipFile(ROOT/'SVR-Depth_实验证据快照_20261109.zip')
for n in z.namelist():
 if not n.endswith('protocol.json') or not ('external_compare' in n or 'v5c_external' in n or 'm04_diagnostic' in n or 'm05_diagnostic' in n):continue
 obj=json.loads(z.read(n));seqs=obj.get('sequence_ids',obj.get('sequences',obj.get('sources')))
 if isinstance(seqs,list):
  for domain in ('TUM','ScanNet','Sintel'):
   if '/'+domain+'/' in n:
    for seq in seqs:known[domain][seq].add('V4/V5 external protocol listing')
 if isinstance(seqs,dict):
  for domain,items in seqs.items():
   for seq in items if isinstance(items,list) else [items]:known[domain][seq].add('M04/M05 diagnostic protocol listing')
rows=[]
for domain,contents in meta.items():
 if 'error' in contents:continue
 splits=contents['splits'];canon={canonical(seq) for seq in known[domain]}
 for x in contents['sequences']:
  seq=x['sequence'];exact=sorted(known[domain].get(seq,[]));overlap=canonical(seq) in canon
  status='KNOWN_DEVELOPMENT_LISTING' if exact else ('SAME_NORMALIZED_NAME_AS_KNOWN_DEV_NOT_CERTIFIED' if overlap else 'UNKNOWN_REQUIRES_FULL_PROVENANCE_AUDIT')
  rows.append({'dataset':domain,'sequence':seq,'split_membership':';'.join(k for k,v in splits.items() if seq in v),'image_count':x['image_count'],'depth_count':x['depth_count'],'metadata_lengths_equal':x['metadata_lengths_equal'],'first16_keys_exist':x['first16_keys_exist'],'exact_known_usage':';'.join(exact),'same_normalized_name_as_known_development':overlap,'candidate_status':status,'provenance_scope':'P2-B manifests and archived V4/V5 external and M04/M05 protocols only; incomplete'})
with (OUT/'independent_candidate_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={d:{'metadata_records':sum(x['dataset']==d for x in rows),'exact_known_development_listing':sum(x['dataset']==d and bool(x['exact_known_usage']) for x in rows),'same_normalized_name_as_known_development':sum(x['dataset']==d and x['same_normalized_name_as_known_development'] for x in rows),'unknown_requires_full_provenance_audit':sum(x['dataset']==d and not x['same_normalized_name_as_known_development'] for x in rows)} for d in meta}
(OUT/'independent_candidate_summary.json').write_text(json.dumps({'datasets':summary,'verified_untouched_independent_candidates':0,'limitation':'Normalized train/val names suggest overlap; exact physical video relation is unverified. History search is incomplete; no candidates certified independent.','Bonn':'P2-B used all five known Bonn development sequences; no additional source scanned.'},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(summary,ensure_ascii=False))
