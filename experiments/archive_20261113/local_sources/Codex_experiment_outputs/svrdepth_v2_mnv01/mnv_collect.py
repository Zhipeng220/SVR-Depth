import csv,json
from pathlib import Path
root=Path("/home/svrdepth/SVR-Depth/velodepth/outputs/svrdepth_v2_mnv01"); base=root/"history_pilot"; rows=[]
for scene in ["Bonn","TUM"]:
 for anchor in [9,14,20]:
  for h in ["H0","H1","H2"]:
   p=base/scene/("anchor_"+str(anchor))/h;f=json.loads((p/"HOLD_features.json").read_text());a=json.loads((p/"HOLD_status.json").read_text());b=json.loads((p/"FULL_status.json").read_text());pre=json.loads((p/"prefix_status.json").read_text());u=a["metrics"]["AbsRel"]-b["metrics"]["AbsRel"];d=f["history_decisions"];full=[x["index"] for x in d if x["mask"]=="1111"];ns=f["history_norms"];r=dict(physical_scene_id="Bonn_person_tracking2" if scene=="Bonn" else "TUM_walking_halfsphere",scene=scene,anchor=anchor,history=h,target=anchor+4,U_absrel=u,U_rmse=a["metrics"]["RMSE"]-b["metrics"]["RMSE"],label="HARMFUL" if u<-.001 else "BENEFICIAL" if u>.001 else "NEUTRAL",hold_absrel=a["metrics"]["AbsRel"],full_absrel=b["metrics"]["AbsRel"],hold_rmse=a["metrics"]["RMSE"],full_rmse=b["metrics"]["RMSE"],valid_pixels=a["metrics"]["valid_pixels"],overall_update_l2=f["overall_update_l2"],overall_relative_norm=f["overall_relative_norm"],history_full_count=len(full),recent5_commit_rate=sum(x["mask"]=="1111" for x in d[-5:])/len(d[-5:]),frames_since_last_full=anchor-full[-1] if full else "",history_hold_count=len(d)-len(full),recent_update_trend=ns[-1]-ns[-2] if len(ns)>1 else "",prefix_steps=pre["run_step_calls"],branch_steps=a["run_step_calls"]+b["run_step_calls"],status="PASS_KEY_TENSORS_MASK_HIDDEN_UNKNOWN")
   for i,v in enumerate(f["layer_update_l2"],1):r["layer%d_update_l2"%i]=v
   for k,v in f["context_parent_before_forward"].items():
    name="rays" if k=="rays_map" else k
    for stat,value in v.items():r[name+"_"+stat]=value
   rows.append(r)
for name in ["MNV01_HISTORY_PILOT.csv","MNV01_HISTORY_UTILITY.csv","MNV01_HISTORY_FEATURES.csv"]:
 with (root/name).open("w",encoding="utf8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print("groups",len(rows),"steps",sum(x["prefix_steps"]+x["branch_steps"] for x in rows),"labels",{k:sum(r["label"]==k for r in rows) for k in ["HARMFUL","BENEFICIAL","NEUTRAL"]},flush=True)
