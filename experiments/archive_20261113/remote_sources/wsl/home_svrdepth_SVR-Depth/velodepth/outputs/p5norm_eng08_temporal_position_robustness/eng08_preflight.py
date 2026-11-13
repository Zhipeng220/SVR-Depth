from pathlib import Path
import csv,json,cv2,numpy as np
root=Path('/home/svrdepth/SVR-Depth/velodepth');out=root/'outputs/p5norm_eng08_temporal_position_robustness';f=root/'outputs/y04_v4_manifest_candidate_20261108_190940/person_tracking2_pairs_v4_candidate.csv'
with f.open(encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h))
checks=[]
for i in range(1,9):
 r=rows[i];assert int(r['frame_index_zero_based'])==i and r['gt_match_status']=='MATCHED';rgb=cv2.imread(r['rgb_path']);gt=cv2.imread(r['depth_path'],cv2.IMREAD_UNCHANGED);assert rgb is not None and gt is not None and rgb.shape[:2]==gt.shape==(480,640) and gt.dtype==np.uint16;mask=np.isfinite(gt)&(gt>0);assert mask.any();checks.append({'index':i,'timestamp':r['rgb_timestamp'],'rgb':r['rgb_path'],'gt':r['depth_path'],'valid_pixels':int(mask.sum())})
assert all(float(checks[i]['timestamp'])<float(checks[i+1]['timestamp']) for i in range(7));(out/'preflight.json').write_text(json.dumps(checks,indent=2));print('PREFLIGHT PASS: frozen frames 1–8')
