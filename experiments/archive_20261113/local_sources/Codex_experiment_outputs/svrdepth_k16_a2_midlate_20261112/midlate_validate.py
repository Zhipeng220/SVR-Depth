from pathlib import Path
import csv,json,numpy as np
p=Path("/home/svrdepth/SVR-Depth/velodepth/outputs/svrdepth_k16_a2_midlate_20261112");t=json.loads((p/"target_map.json").read_text());rows=[]
def rd(f):
 with f.open(newline="") as h:return list(csv.DictReader(h))
for name in ["balloon_tracking","crowd3"]:
 for cycle in t[name]:
  st=cycle*16;end=st+15
  for arm in ["O","P"]:
   a=p/"trajectories"/(name+"_"+arm+"_continuous_0");b=p/"trajectories"/(name+"_"+arm+"_short_"+str(st));checks={}
   for frame in [st,st+1]:
    x=np.load(a/("state_%04d.npz"%frame));y=np.load(b/("state_%04d.npz"%frame));checks["keyframe_state" if frame==st else "first_commit_state"]=set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files);checks["keyframe_decoder" if frame==st else "first_commit_decoder"]=json.loads((a/("decoder_%04d.json"%frame)).read_text())==json.loads((b/("decoder_%04d.json"%frame)).read_text())
   x=np.load(a/("head_%04d.npz"%(st+1)));y=np.load(b/("head_%04d.npz"%(st+1)));checks["repair_head"]=set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
   checks["predictions_16"]=all(np.array_equal(np.load(a/("pred_%04d.npy"%i)),np.load(b/("pred_%04d.npy"%i))) for i in range(st,end+1))
   aa={int(x["frame"]):x for x in rd(a/"frames.csv")};bb={int(x["frame"]):x for x in rd(b/"frames.csv")};checks["mask_metric_16"]=all(aa[i]["status"]==bb[i]["status"] and aa[i]["mask_sha256"]==bb[i]["mask_sha256"] and aa[i]["AbsRel"]==bb[i]["AbsRel"] for i in range(st,end+1))
   sa=next(x for x in rd(a/"scores.csv") if int(x["cycle_id"])==cycle);sb=next(x for x in rd(b/"scores.csv") if int(x["cycle_id"])==cycle);checks["layer1"]=sa["update_l1"]==sb["update_l1"];rows.append(dict(sequence=name,cycle_id=cycle,arm=arm,start_frame=st,end_frame=end,**checks,status="PASS" if all(checks.values()) else "FAIL"))
with (p/"SHORT_REPLAY_VALIDATION.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print("checks",len(rows),"PASS",sum(x["status"]=="PASS" for x in rows));print([x for x in rows if x["status"]!="PASS"]);assert len(rows)==8 and all(x["status"]=="PASS" for x in rows)
