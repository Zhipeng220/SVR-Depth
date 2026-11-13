import os,sys,csv,json,io,re
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
import numpy as np,h5py
from PIL import Image
r=Path("/home/svrdepth/SVR-Depth/velodepth");p=r/"outputs/mnv02_raw_frame_recovery";p.mkdir(exist_ok=False);(p/"work").mkdir();(p/"MNV02_MATCH_VISUALIZATIONS").mkdir()
def save(name,rows):
 with (p/name).open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
inv=[];ident=[]
for ds in ["TUM","ScanNet","Sintel"]:
 path=Path("/home/svrdepth/datasets/velodepth_evaluation")/(ds+".hdf5")
 with h5py.File(path,"r") as h:
  keys=[k for k in h.keys()];key="sequences_.json" if ds=="ScanNet" else "sequences.json";data=json.loads(np.asarray(h[key]).tobytes().decode());
  for sid,x in sorted(data.items()):
   images=x.get("image",[]);depth=x.get("depth",[]);row=dict(dataset=ds,sequence_id=sid,frame_count=len(images),depth_count=len(depth),rgb_shape="UNKNOWN",rgb_dtype="UNKNOWN",depth_shape="UNKNOWN",depth_dtype="UNKNOWN",depth_sample_min="UNKNOWN",depth_sample_max="UNKNOWN",sample_valid_ratio="UNKNOWN",camera_fields=";".join(k for k in ["K","camera_params","camera_model","cam2w"] if k in x),timestamp_fields=";".join(k for k in x if "timestamp" in k.lower()) or "UNKNOWN",source_frame_mapping="HDF5 path strings only",attributes="root="+str(dict(h.attrs)),top_level_key_count=len(keys),inspection_scope="first frame sample only")
   try:
    a=np.asarray(Image.open(io.BytesIO(np.asarray(h[images[0]]).tobytes())));b=np.asarray(Image.open(io.BytesIO(np.asarray(h[depth[0]]).tobytes())));b=(b[:,:,0].astype(np.int32)|(b[:,:,1].astype(np.int32)<<8)|(b[:,:,2].astype(np.int32)<<16)) if b.ndim==3 else b;row.update(rgb_shape=str(a.shape),rgb_dtype=str(a.dtype),depth_shape=str(b.shape),depth_dtype=str(b.dtype),depth_sample_min=int(b.min()),depth_sample_max=int(b.max()),sample_valid_ratio=float((np.isfinite(b)&(b>0)).mean()))
   except Exception as e:row["inspection_scope"]="SAMPLE_ERROR: "+str(e)
   inv.append(row)
   if ds=="ScanNet":
    m=re.search(r"scene(\d{4})_(\d{2})",sid)
    if m:ident.append(dict(sequence_id=sid,space_id=m.group(1),scan_id=m.group(2),physical_location_group="space_"+m.group(1),identity_basis="official naming; visual and exposure still require review",raw_sens="NOT_FOUND_IN_CHECKED_DIRECTORIES",status="BLOCKED_AUTHORIZATION"))
save("MNV02_HDF5_SEQUENCE_INVENTORY.csv",inv);save("MNV02_SCANNET_SCENE_IDENTITY.csv",ident)
print("counts",{d:sum(x["dataset"]==d for x in inv) for d in ["TUM","ScanNet","Sintel"]},"spaceids",len(set(x["space_id"] for x in ident)),flush=True)
