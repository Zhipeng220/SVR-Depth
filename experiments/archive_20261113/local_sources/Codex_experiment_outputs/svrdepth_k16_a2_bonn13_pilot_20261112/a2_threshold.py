import csv,math,numpy as np
from pathlib import Path
p=Path("/home/svrdepth/SVR-Depth/velodepth/outputs/svrdepth_k16_a2_bonn13_pilot_20261112");r=list(csv.DictReader((p/"BONN13_K16_PILOT_CYCLE_RESULTS.csv").open()));seq=sorted(set(x["sequence"] for x in r))
def evaluate(rows,t):
 vals=[];npicks=0
 for name in sorted(set(x["sequence"] for x in rows)):
  q=[x for x in rows if x["sequence"]==name];w=np.array([float(x["valid_frame_count"]) for x in q]);a=np.array([float(x["update_l1"])>t for x in q]);npicks+=int(a.sum());vals.append(float(np.average([float(x["error_P"] if y else x["error_O"]) for x,y in zip(q,a)],weights=w)))
 return float(np.mean(vals)),npicks
def fit(rows):
 v=sorted(set(float(x["update_l1"]) for x in rows));ts=[-math.inf]+[(a+b)/2 for a,b in zip(v,v[1:])]+[math.inf];z=[(evaluate(rows,t)[0],evaluate(rows,t)[1],t) for t in ts];return min(z),z
(best,grid)=fit(r);out=[dict(theta=str(t),mean_absrel=e,P_cycle_count=n,role="PILOT_THRESHOLD_NOT_FINAL") for e,n,t in grid]
with (p/"BONN13_PILOT_THRESHOLD_SWEEP.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
loo=[]
for name in seq:
 tr=[x for x in r if x["sequence"]!=name];te=[x for x in r if x["sequence"]==name];(e,n,t),_=fit(tr);test,k=evaluate(te,t);eo,_=evaluate(te,math.inf);ep,_=evaluate(te,-math.inf);loo.append(dict(heldout_sequence=name,theta=str(t),test_absrel=test,test_O=eo,test_P=ep,test_P_cycles=k,training_P_cycles=n,degenerate_train_rule=not math.isfinite(t),role="PILOT_NOT_FINAL"))
with (p/"BONN13_PILOT_LOO.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=loo[0]);w.writeheader();w.writerows(loo)
print("pilot optimum",best,"LOO degenerate",sum(x["degenerate_train_rule"] for x in loo),"LOO avg",np.mean([x["test_absrel"] for x in loo]))
