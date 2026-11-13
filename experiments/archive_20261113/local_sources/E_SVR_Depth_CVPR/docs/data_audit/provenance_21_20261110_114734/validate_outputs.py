import csv,json,re
from pathlib import Path
root=Path.cwd();p=root/'docs/data_audit/provenance_21_20261110_114734'
def read(n):
 with (p/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
selected=json.loads((p/'selected_sequences.json').read_text(encoding='utf8'));m=read('physical_provenance_mapping.csv');f=read('frame_inventory.csv');t=read('temporal_overlap_report.csv');rel=read('physical_relationships.csv');lead=read('known_tum_depth_lead.csv')
assert len(m)==21 and len({(x['dataset'],x['selected_sequence_id']) for x in m})==21
assert {(x['dataset'],x['selected_sequence_id']) for x in m}=={(x['dataset'],x['sequence']) for x in selected['sequences']}
assert all(x['P2B']==x['P3A']==x['P3B']=='YES' and x['physical_recording_relation']=='UNKNOWN' for x in m)
assert len(f)==9432 and len({(x['dataset'],x['sequence'],x['index'],x['modality']) for x in f})==len(f)
assert all(re.fullmatch('[0-9a-f]{64}',x['sha256']) and re.fullmatch('[0-9a-f]{64}',x['decoded_pixel_sha256']) and re.fullmatch('[0-9a-f]{16}',x['phash64']) for x in f)
assert len(t)==420 and len({(x['dataset_a'],x['sequence_a'],x['dataset_b'],x['sequence_b'],x['modality']) for x in t})==420
assert len(rel)==211 and all(x['recording_relation']=='UNKNOWN' and x['session_relation']=='UNKNOWN' for x in rel)
assert len(lead)==4 and lead[0]['sha256']==lead[1]['sha256'] and lead[2]['sha256']!=lead[3]['sha256']
print('VALIDATION_PASS 21 sequences 42 anchors 9432 frame records 420 sequence-pair/modality comparisons; known TUM lead checked')

