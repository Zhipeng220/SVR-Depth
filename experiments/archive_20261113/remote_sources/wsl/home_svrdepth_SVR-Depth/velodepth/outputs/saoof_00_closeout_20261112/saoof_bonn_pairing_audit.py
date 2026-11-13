"""Audit HDF5/raw Bonn sample correspondence; candidate times are not proof."""
import csv, json, re
from pathlib import Path
import cv2, h5py, numpy as np

root=Path('/home/svrdepth/datasets/bonn/rgbd_bonn_person_tracking2')
out=Path('outputs/saoof_00_closeout_20261112'); out.mkdir(exist_ok=True,parents=True)
def entries(name):
 return [(float(t),root/p) for line in (root/name).read_text().splitlines() if line.strip() and not line.startswith('#') for t,p in [line.split()[:2]]]
rgb=entries('rgb.txt'); dep=entries('depth.txt')
rows=[]
with h5py.File('/home/svrdepth/datasets/velodepth_evaluation/Bonn.hdf5','r') as h:
 sequences=json.loads(h['sequences.json'][()].tobytes().decode('ascii'))
 key=next(k for k in sequences if k.endswith('/person_tracking2_val')); s=sequences[key]
 for i in range(78,94):
  ip=s['image'][i]; dp=s['depth'][i]
  stem=Path(ip).stem.removesuffix('_c'); m=re.fullmatch(r'(\d+)-(\d+)',stem)
  if not m: raise ValueError(stem)
  fragment=float(m.group(1)+'.'+m.group(2))
  # Filenames contain truncated time-like suffix; modulo and nearest match are hypotheses.
  candidates=sorted(rgb,key=lambda x:abs((x[0]%10000)-fragment))
  t,rp=candidates[0]; rawrgb=cv2.imread(str(rp),cv2.IMREAD_COLOR)
  image=cv2.imdecode(np.asarray(h[ip]),cv2.IMREAD_COLOR)
  dt,rd=min(dep,key=lambda x:abs(x[0]-t)); rawdepth=cv2.imread(str(rd),cv2.IMREAD_UNCHANGED)
  depth=cv2.imdecode(np.asarray(h[dp]),cv2.IMREAD_UNCHANGED)
  same_rgb=rawrgb.shape==image.shape
  same_d=rawdepth.shape==depth.shape
  valid=(rawdepth>0)&(depth>0) if same_d else None
  rows.append({'hdf_index':i,'hdf_rgb':ip,'hdf_depth':dp,'time_fragment_hypothesis':fragment,'rgb_candidates_within_5ms':sum(abs((x[0]%10000)-fragment)<=.0051 for x in rgb),'raw_rgb_candidate':str(rp),'raw_rgb_time':f'{t:.5f}','candidate_time_difference_ms':round(abs((t%10000)-fragment)*1000,3),'raw_nearest_depth_candidate':str(rd),'raw_depth_time':f'{dt:.5f}','raw_rgb_depth_delta_ms':round(abs(t-dt)*1000,3),'hdf_rgb_shape':str(image.shape),'raw_rgb_shape':str(rawrgb.shape),'hdf_depth_shape':str(depth.shape),'raw_depth_shape':str(rawdepth.shape),'rgb_bgr_mae':round(float(np.mean(abs(image.astype(np.float32)-rawrgb.astype(np.float32)))),5) if same_rgb else '', 'hdf_depth_dtype':str(depth.dtype),'raw_depth_dtype':str(rawdepth.dtype),'depth_both_valid_pixels':int(valid.sum()) if same_d else '', 'depth_median_hdf_mm_vs_raw_5000_m_difference_m':float(np.median(abs(depth[valid]/1000.-rawdepth[valid]/5000.))) if same_d and valid.any() else '', 'camera_model':str(s['camera_model'][i]),'camera_params':json.dumps(s['camera_params'][i]),'timestamp_association':'candidate only; official source timestamp and conversion not verified'})
  rows[-1]['rgb_candidates_within_10ms']=sum(abs((x[0]%10000)-fragment)<=.0101 for x in rgb)
  if same_d and valid.any():
   dd=np.abs(depth[valid]/1000.-rawdepth[valid]/5000.)
   rows[-1]['positive_validity_mismatch_pixels']=int(((depth>0)!=(rawdepth>0)).sum())
   rows[-1]['depth_diff_p95_m']=float(np.percentile(dd,95))
   rows[-1]['depth_diff_mean_m']=float(dd.mean())
   rows[-1]['depth_diff_within_1mm_fraction']=float((dd<=.001001).mean())
with (out/'bonn_pairing_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r['hdf_index'],r['hdf_rgb'],r['raw_rgb_time'],r['candidate_time_difference_ms'],r['rgb_bgr_mae'],r['raw_rgb_depth_delta_ms'],r['depth_median_hdf_mm_vs_raw_5000_m_difference_m'])
