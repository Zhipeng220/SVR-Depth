"""Cross-implementation full-frame ray diagnostics, not physical calibration."""
from pathlib import Path
import json,cv2,numpy as np,torch
from velodepth.utils.camera import OPENCV
K=np.array([[542.822841,0,315.593520],[0,542.576870,237.756098],[0,0,1]],np.float64)
d=np.array([.039903,-.099343,-.000730,-.000144,0.],np.float64)
out=Path('outputs/saoof_00_contract_v2_20261112'); results=[]
for h,w in [(480,640),(434,574),(224,294),(294,392)]:
 yy,xx=np.mgrid[:h,:w].astype(np.float64); sx=w/640; sy=h/480
 x=(xx+.5)/sx-.5; y=(yy+.5)/sy-.5; valid=(x>=0)&(x<=639)&(y>=0)&(y<=479)
 uv=np.stack([x[valid],y[valid]],axis=1); refxy=cv2.undistortPoints(uv.reshape(-1,1,2),K,d).reshape(-1,2)
 ref=np.column_stack((refxy,np.ones(len(refxy)))); ref/=np.linalg.norm(ref,axis=1,keepdims=True)
 for ordering in ('direct','swapped'):
  p0,p1=(d[2],d[3]) if ordering=='direct' else (d[3],d[2])
  pars=torch.tensor([[K[0,0],K[1,1],K[0,2],K[1,2],d[0],d[1],d[4],0,0,0,p0,p1,0,0,0,0]],dtype=torch.float32)
  camera=OPENCV(params=pars); pieces=[]
  with torch.no_grad():
   for part in np.array_split(uv,max(1,int(np.ceil(len(uv)/32768)))):
    t=torch.as_tensor(part.T,dtype=torch.float32).reshape(1,2,1,-1)
    pieces.append(camera.unproject(t)[0,:,0,:].T.numpy())
  ray=np.concatenate(pieces).astype(np.float64); ray/=np.linalg.norm(ray,axis=1,keepdims=True)
  angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(ray,ref),axis=1),np.sum(ray*ref,axis=1)))
  edge=(uv[:,0]<32)|(uv[:,0]>607)|(uv[:,1]<24)|(uv[:,1]>455)
  results.append({'size':[h,w],'ordering':ordering,'n':len(uv),'median_deg':float(np.median(angle)),'p95_deg':float(np.percentile(angle,95)),'max_deg':float(angle.max()),'edge_p95_deg':float(np.percentile(angle[edge],95)),'edge_max_deg':float(angle[edge].max())})
  print(results[-1],flush=True)
(out/'camera_crosscheck.json').write_text(json.dumps({'results':results,'caveat':'OpenCV undistort vs project OPENCV.unproject numerical comparison; not physical alignment or native predicted-camera equivalence. No padding present in inspected sequence.'},indent=2),encoding='utf8')
