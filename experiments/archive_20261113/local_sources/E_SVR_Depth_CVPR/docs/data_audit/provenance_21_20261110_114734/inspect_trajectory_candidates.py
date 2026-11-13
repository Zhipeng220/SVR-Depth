import csv,io,itertools,json,zipfile
from pathlib import Path
root=Path.cwd();p=root/'docs/data_audit/provenance_21_20261110_114734';z=root/'y04_v4_physical_scene_audit_20261108_codex.zip'
with zipfile.ZipFile(z) as q:
 name=next(x for x in q.namelist() if x.endswith('/sequence_metadata.csv'));rows=list(csv.DictReader(io.StringIO(q.read(name).decode('utf-8-sig'))))
selected=json.loads((p/'selected_sequences.json').read_text(encoding='utf8'))['sequences'];names={x['sequence'] for x in selected if x['dataset']=='Bonn'}|{'crowd3'};chosen=[r for r in rows if r['sequence'] in names]
if len(chosen)!=6:raise RuntimeError('Bonn trajectory metadata incomplete')
out=[]
for a,b in itertools.combinations(chosen,2):
 lo1=json.loads(a['gt_xyz_min']);hi1=json.loads(a['gt_xyz_max']);lo2=json.loads(b['gt_xyz_min']);hi2=json.loads(b['gt_xyz_max']);over=[max(0.,min(hi1[i],hi2[i])-max(lo1[i],lo2[i])) for i in range(3)]
 out.append({'sequence_a':a['sequence'],'sequence_b':b['sequence'],'trajectory_path_a':a['groundtruth_path'],'trajectory_path_b':b['groundtruth_path'],'a_xyz_min':json.dumps(lo1),'a_xyz_max':json.dumps(hi1),'b_xyz_min':json.dumps(lo2),'b_xyz_max':json.dumps(hi2),'xyz_bbox_overlap_dimensions':sum(x>0 for x in over),'bbox_overlap_lengths':json.dumps(over),'coordinate_frame_verified':'NO','interpretation':'UNVERIFIED coordinate frame; descriptive candidate only, cannot infer shared scene, recording, session or independence'})
with (p/'trajectory_bbox_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print('TRAJECTORY_CANDIDATES',len(out))

