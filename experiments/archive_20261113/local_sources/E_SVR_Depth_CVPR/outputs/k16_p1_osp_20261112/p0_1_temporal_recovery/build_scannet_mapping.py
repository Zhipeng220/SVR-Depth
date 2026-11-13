import csv,json
from pathlib import Path
from collections import Counter
import h5py
out=Path(__file__).resolve().parent
with (out.parent/'scannet_gt_candidates.csv').open(encoding='utf-8-sig',newline='') as f: gt={r['sequence_id']:r for r in csv.DictReader(f)}
fields=['domain','sequence_id','split','candidate_scan_id','candidate_space_id','hdf5_frames','rgb_key_first','depth_key_first','rgb_key_last','depth_key_last','same_key_stem_count','numeric_rgb_stem_count','rgb_stem_delta_counts','rgb_timestamp','depth_timestamp','rgb_gt_time_delta','temporal_continuity_certified','rgb_gt_temporal_pair_certified','physical_independence_certified','gt_complete_index_candidates','classification','evidence']
frame_fields=['sequence_id','hdf5_index','rgb_key','depth_key','rgb_export_label','depth_export_label','rgb_label_delta_from_previous','depth_label_delta_from_previous','rgb_original_frame_id','depth_original_frame_id','rgb_timestamp','depth_timestamp','rgb_depth_time_delta','temporal_status']
summaries=[]
with h5py.File(r'D:\SVR-Depth-data\VeloDepth-Evaluation\ScanNet.hdf5','r') as h, (out/'P01_SCANNET_FRAME_KEYS.csv').open('w',encoding='utf-8-sig',newline='') as fw:
 seq=json.loads(bytes(h['sequences_.json'][:]).decode('ascii')); writer=csv.DictWriter(fw,fieldnames=frame_fields); writer.writeheader()
 for name,item in sorted(seq.items()):
  images=item['image']; depths=item['depth']; prev_rgb=prev_dep=None; deltas=[]; paired=0; numeric=0
  for i,(rgb,dep) in enumerate(zip(images,depths)):
   r=Path(rgb).stem; d=Path(dep).stem; ri=int(r) if r.isdigit() else None; di=int(d) if d.isdigit() else None
   rd=ri-prev_rgb if ri is not None and prev_rgb is not None else None; dd=di-prev_dep if di is not None and prev_dep is not None else None
   if rd is not None: deltas.append(rd)
   numeric += ri is not None; paired += r==d
   writer.writerow(dict(sequence_id=name,hdf5_index=i,rgb_key=rgb,depth_key=dep,rgb_export_label=r,depth_export_label=d,rgb_label_delta_from_previous=rd if rd is not None else '',depth_label_delta_from_previous=dd if dd is not None else '',rgb_original_frame_id='UNKNOWN',depth_original_frame_id='UNKNOWN',rgb_timestamp='UNKNOWN',depth_timestamp='UNKNOWN',rgb_depth_time_delta='UNKNOWN',temporal_status='UNKNOWN: key labels are not verified raw frame IDs'))
   prev_rgb,prev_dep=ri,di
  scan=name.rsplit('_',1)[0]; space=scan.rsplit('_',1)[0]; split=name.rsplit('_',1)[-1]; g=gt.get(name)
  summaries.append(dict(domain='ScanNet',sequence_id=name,split=split,candidate_scan_id=scan,candidate_space_id=space,hdf5_frames=len(images),rgb_key_first=images[0] if images else '',depth_key_first=depths[0] if depths else '',rgb_key_last=images[-1] if images else '',depth_key_last=depths[-1] if depths else '',same_key_stem_count=paired,numeric_rgb_stem_count=numeric,rgb_stem_delta_counts=';'.join(f'{k}:{v}' for k,v in sorted(Counter(deltas).items())),rgb_timestamp='UNKNOWN',depth_timestamp='UNKNOWN',rgb_gt_time_delta='UNKNOWN',temporal_continuity_certified='NO',rgb_gt_temporal_pair_certified='NO',physical_independence_certified='NO',gt_complete_index_candidates=g['gt_complete_index_cycle_count'] if g else 'UNKNOWN',classification='D: index GT candidates only; E: original provenance insufficient' if g and int(g['gt_complete_index_cycle_count']) else 'E: original provenance insufficient',evidence='sequences_.json image/depth lists; no original .sens or verified conversion mapping'))
with (out/'P01_SCANNET_TEMPORAL_MAPPING.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summaries)
print('sequences',len(summaries),'frames',sum(x['hdf5_frames'] for x in summaries),'same-stems',sum(x['same_key_stem_count'] for x in summaries))
