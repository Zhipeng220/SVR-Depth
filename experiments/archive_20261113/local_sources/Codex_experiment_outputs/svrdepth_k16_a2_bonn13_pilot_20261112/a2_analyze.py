from pathlib import Path
import csv,json,statistics,math
import numpy as np
from scipy.stats import spearmanr
p=Path("/home/svrdepth/SVR-Depth/velodepth/outputs/svrdepth_k16_a2_bonn13_pilot_20261112")
def rd(f):
 with f.open(encoding="utf8",newline="") as h:return list(csv.DictReader(h))
def wr(f,r):
 with (p/f).open("w",encoding="utf8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
seq=rd(p/"BONN13_SEQUENCE_MANIFEST.csv");frames=[];cycles=[];summary=[];total=[];comparison=[]
for q in seq:
 name=q["sequence"]; dirs={a:p/"trajectories"/(name+"_"+a) for a in ["O","P"]};
 if not all((d/"status.json").exists() and json.loads((d/"status.json").read_text())["status"]=="PASS" for d in dirs.values()):continue
 f={a:rd(dirs[a]/"frames.csv") for a in dirs};score={a:rd(dirs[a]/"scores.csv") for a in dirs};assert all(len(f[a])==64 and len(score[a])==4 for a in dirs);frames+=f["O"]+f["P"]; local=[]
 for k in range(4):
  o=f["O"][16*k:16*k+16];v=f["P"][16*k:16*k+16];so=score["O"][k];sp=score["P"][k];assert all(x["frame"]==y["frame"] and x["status"]==y["status"] and x["mask_sha256"]==y["mask_sha256"] for x,y in zip(o,v));valid=[(x,y) for x,y in zip(o,v) if x["status"]=="OK"];act=[(x,y) for x,y in valid if int(x["age"])>=2]; delta=float(so["update_l1"])-float(sp["update_l1"]);assert delta==0 and so["input_shape"]==sp["input_shape"] and so["repair_mode"]==sp["repair_mode"]=="pre_readout";assert len(valid)>0 and len(act)>0
  eo=np.mean([float(x["AbsRel"]) for x,y in valid]);ep=np.mean([float(y["AbsRel"]) for x,y in valid]);u=eo-ep;ua=np.mean([float(x["AbsRel"])-float(y["AbsRel"]) for x,y in act]);r=dict(sequence=name,physical_scene_id="UNKNOWN",cycle_id=k,start_frame=16*k,end_frame=16*k+15,decision_frame=16*k+1,update_l1_O=so["update_l1"],update_l1_P=sp["update_l1"],update_l1=so["update_l1"],score_difference=delta,update_overall=so["update_overall"],feature_shape=so["input_shape"],num_elements=so["num_elements"],dtype=so["dtype"],valid_frame_count=len(valid),actionable_frame_count=len(act),error_O=eo,error_P=ep,utility_cycle=u,utility_action=ua,preferred_action="P" if u>0 else "O" if u<0 else "TIE",label_provenance="real fixed O/P; A1 conditional cycle eligibility",eligibility_status="PASS");cycles.append(r);local.append(r)
 w=np.array([x["valid_frame_count"] for x in local]);eo=np.average([x["error_O"] for x in local],weights=w);ep=np.average([x["error_P"] for x in local],weights=w);oracle=np.average([min(x["error_O"],x["error_P"]) for x in local],weights=w);summary.append(dict(sequence=name,cycles=4,O_wins=sum(x["preferred_action"]=="O" for x in local),P_wins=sum(x["preferred_action"]=="P" for x in local),ties=sum(x["preferred_action"]=="TIE" for x in local),O_absrel=eo,P_absrel=ep,period_oracle=oracle,oracle_gain=min(eo,ep)-oracle))
assert len(cycles)==52
wr("BONN13_K16_PILOT_FRAME_RESULTS.csv",frames);wr("BONN13_K16_PILOT_CYCLE_RESULTS.csv",cycles);wr("BONN13_SEQUENCE_SELECTION_SPACE.csv",summary)
O=np.mean([x["O_absrel"] for x in summary]);P=np.mean([x["P_absrel"] for x in summary]);V=np.mean([min(x["O_absrel"],x["P_absrel"]) for x in summary]);C=np.mean([x["period_oracle"] for x in summary]);wr("BONN13_ORACLE_SUMMARY.csv",[dict(sequences=len(summary),cycles=len(cycles),O=O,P=P,per_video_oracle=V,cycle_oracle=C,gain_vs_global_fixed=min(O,P)-C,gain_within_video=V-C,offline_oracle_only=True)])
u=np.array([x["utility_cycle"] for x in cycles]);wr("BONN13_UTILITY_DISTRIBUTION.csv",[dict(count=len(u),mean=u.mean(),median=np.median(u),p10=np.quantile(u,.1),p25=np.quantile(u,.25),p75=np.quantile(u,.75),p90=np.quantile(u,.9),O_wins=int((u<0).sum()),P_wins=int((u>0).sum()),ties=int((u==0).sum()),mixed_sequences=sum(x["O_wins"]>0 and x["P_wins"]>0 for x in summary),O_win_error_sum=sum((-x["utility_cycle"])*x["valid_frame_count"] for x in cycles if x["utility_cycle"]<0),P_win_error_sum=sum(x["utility_cycle"]*x["valid_frame_count"] for x in cycles if x["utility_cycle"]>0),error_sum_units="sum per-frame AbsRel difference; not independent samples")])
D=[]
for feature in ["update_l1","update_overall"]:
 for target in ["utility_cycle","utility_action"]:
  for name,sub in [("ALL",cycles)]+[(s["sequence"],[x for x in cycles if x["sequence"]==s["sequence"]]) for s in summary]:
   xx=[float(x[feature]) for x in sub];yy=[float(x[target]) for x in sub];rho=float(spearmanr(xx,yy).statistic) if len(set(xx))>1 and len(set(yy))>1 else float("nan");D.append(dict(scope=name,feature=feature,target=target,n=len(sub),spearman=rho,interpretation="exploratory four points per video; cycles not independent scenes"))
wr("BONN13_LAYER1_DIAGNOSTICS.csv",D);(p/"analysis.json").write_text(json.dumps(dict(O=O,P=P,video_oracle=V,cycle_oracle=C,gain_within=V-C,gain_global=min(O,P)-C,O_wins=int((u<0).sum()),P_wins=int((u>0).sum()),ties=int((u==0).sum()),mixed=sum(x["O_wins"]>0 and x["P_wins"]>0 for x in summary)),indent=2));print((p/"analysis.json").read_text())
