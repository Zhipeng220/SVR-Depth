from pathlib import Path
import csv,numpy as np,json
p=Path("/home/svrdepth/SVR-Depth/velodepth/outputs/svrdepth_k16_a2_midlate_20261112")
def read(f):
 with f.open(newline="") as h:return list(csv.DictReader(h))
def save(n,rows):
 with (p/n).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
target=read(p/"MIDLATE_TARGET_MANIFEST.csv");actions={(x["sequence"],int(x["cycle_id"])):x for x in read(p/"MIDLATE_FROZEN_ACTIONS.csv")};cycles=[];frames=[]
for t in target:
 n=t["sequence"];k=int(t["cycle_id"]);st=int(t["start_frame"]);act=actions[(n,k)]; z={}
 for arm in ["O","P"]:z[arm]=read(p/"trajectories"/(n+"_"+arm+"_short_"+str(st))/"frames.csv");frames+=z[arm]
 assert all(len(z[a])==16 for a in z);assert all(x["frame"]==y["frame"] and x["status"]==y["status"] and x["mask_sha256"]==y["mask_sha256"] for x,y in zip(z["O"],z["P"]));v=[(x,y) for x,y in zip(z["O"],z["P"]) if x["status"]=="OK"];q=[(x,y) for x,y in v if int(x["age"])>=2];assert v and q;eo=float(np.mean([float(x["AbsRel"]) for x,y in v]));ep=float(np.mean([float(y["AbsRel"]) for x,y in v]));ua=float(np.mean([float(x["AbsRel"])-float(y["AbsRel"]) for x,y in q]));cycles.append(dict(sequence=n,physical_scene_id="UNKNOWN",cycle_id=k,position_group=t["position_group"],start_frame=st,decision_frame=st+1,score=act["score"],frozen_action=act["action"],nonfinite_fallback=act["nonfinite_fallback"],valid_frames=len(v),actionable_frames=len(q),E_O=eo,E_P=ep,U_cycle=eo-ep,U_action=ua))
save("BONN13_MIDLATE_FRAME_RESULTS.csv",frames);save("BONN13_MIDLATE_CYCLE_RESULTS.csv",cycles);summary=[]
for n in dict.fromkeys(x["sequence"] for x in cycles):
 a=next(x for x in cycles if x["sequence"]==n and x["position_group"]=="middle");b=next(x for x in cycles if x["sequence"]==n and x["position_group"]=="late");w=np.array([a["valid_frames"],b["valid_frames"]]);weighted=lambda v:float(np.average(v,weights=w));Eo=weighted([a["E_O"],b["E_O"]]);Ep=weighted([a["E_P"],b["E_P"]]);Et=weighted([x["E_P"] if x["frozen_action"]=="P" else x["E_O"] for x in [a,b]]);k=sum(x["frozen_action"]=="P" for x in [a,b]);Er=Ep if k==2 else Eo if k==0 else (weighted([a["E_P"],b["E_O"]])+weighted([a["E_O"],b["E_P"]]))/2;oracle=weighted([min(a["E_O"],a["E_P"]),min(b["E_O"],b["E_P"])]);summary.append(dict(sequence=n,group="A" if n in ["balloon_tracking","removing_obstructing_box"] else "B",middle_U=a["U_cycle"],late_U=b["U_cycle"],middle_score=a["score"],late_score=b["score"],actions=a["frozen_action"]+"/"+b["frozen_action"],valid_middle=a["valid_frames"],valid_late=b["valid_frames"],always_O=Eo,always_P=Ep,random_expectation=Er,threshold=Et,oracle=oracle,G_vs_P=Ep-Et,G_timing=Er-Et,mixed_selection=k==1,heterogeneous_utility=a["U_cycle"]*b["U_cycle"]<0))
save("MIDLATE_SEQUENCE_COMPARISON.csv",summary);groups=[]
for g in ["A","B","ALL"]:
 q=[x for x in summary if g=="ALL" or x["group"]==g];groups.append(dict(group=g,sequence_count=len(q),mean_O=np.mean([x["always_O"] for x in q]),mean_P=np.mean([x["always_P"] for x in q]),mean_random=np.mean([x["random_expectation"] for x in q]),mean_threshold=np.mean([x["threshold"] for x in q]),mean_oracle=np.mean([x["oracle"] for x in q]),G_vs_P=np.mean([x["G_vs_P"] for x in q]),G_timing=np.mean([x["G_timing"] for x in q]),mixed_videos=sum(x["mixed_selection"] for x in q),heterogeneous_videos=sum(x["heterogeneous_utility"] for x in q),contribution_vs_P_to_overall=sum(x["G_vs_P"] for x in q)/13,contribution_timing_to_overall=sum(x["G_timing"] for x in q)/13))
save("MIDLATE_GROUP_COMPARISON.csv",groups);print("cycles",len(cycles),"P",sum(x["frozen_action"]=="P" for x in cycles),"groups",groups)
