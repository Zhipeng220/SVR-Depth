from pathlib import Path
import csv,json,re,h5py
base=Path('/home/svrdepth/SVR-Depth/velodepth/outputs');source=base/'p5norm_v2_4_independent_replication';out=base/'p5norm_eng07r_negative_utility';out.mkdir(exist_ok=False)
def rd(path):
 with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
labels=rd(source/'P5NORM_V2_4_UTILITY_LABELS.csv');anchors=rd(source/'P5NORM_V2_4_FROZEN_ANCHORS.csv');candidates=[x for x in labels if x['dataset']=='ScanNet' and float(x['U_AbsRel'])<0];allowed={'scene0708_00_val','scene0710_00_val','scene0712_00_val','scene0713_00_val'};assert len(candidates)==7 and all(x['sequence_id'] in allowed for x in candidates)
trace=[];summary=[]
with h5py.File(Path.home()/'datasets/velodepth_evaluation/ScanNet.hdf5','r') as h:
 md=json.loads(bytes(h['sequences_.json'][:]).decode())
 for order,c in enumerate(candidates,1):
  seq=c['sequence_id'];anchor=int(c['anchor_idx']);a=next(x for x in anchors if x['dataset']=='ScanNet' and x['sequence_id']==seq and int(x['anchor_idx'])==anchor);series=md[seq];indices=[];good=True
  for i in range(anchor,anchor+5):
   rgb=series['image'][i];depth=series['depth'][i];ri=int(Path(rgb).stem);di=int(Path(depth).stem);indices.append(ri);both=rgb in h and depth in h;good=good and both and ri==di
   trace.append(dict(candidate_order=order,sequence_id=seq,anchor_idx=anchor,stream_index=i,source_frame_id=ri,rgb_hdf5_key=rgb,depth_hdf5_key=depth,rgb_gt_same_source_id=(ri==di),rgb_and_gt_key_exists=both,source_timestamp='UNKNOWN',timestamp_source='NOT FOUND',sample_protocol='UNKNOWN',status='NO-GO: raw timestamp absent'))
  snap=source/'runs'/'ScanNet'/seq/'hold'/('anchor_%d_tensors.pt'%anchor);summary.append(dict(order=order,sequence=seq,anchor=anchor,utility=float(c['U_AbsRel']),source_indices=indices,raw_index_deltas=[b-a for a,b in zip(indices,indices[1:])],list_adjacent=True,rgb_gt_ids_match=good,anchor_timestamp=a['anchor_timestamp'],target_timestamp=a['target_timestamp'],hook_file_exists=snap.exists(),eligible=False))
with (out/'P5NORM_ENG07R_TIMESTAMP_TRACE.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
(out/'stagea_evidence.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
