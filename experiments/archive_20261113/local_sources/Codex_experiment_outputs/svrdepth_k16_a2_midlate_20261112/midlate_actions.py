from pathlib import Path
import csv,math,json
p=Path("/home/svrdepth/SVR-Depth/velodepth/outputs/svrdepth_k16_a2_midlate_20261112");targets=list(csv.DictReader((p/"MIDLATE_TARGET_MANIFEST.csv").open()));out=[]
for t in targets:
 n=t["sequence"];k=int(t["cycle_id"]);st=int(t["start_frame"]);s={}
 for a in ["O","P"]:
  d=p/"trajectories"/(n+"_"+a+"_short_"+str(st));status=json.loads((d/"status.json").read_text());assert status["status"]=="PASS";v=list(csv.DictReader((d/"scores.csv").open()));assert len(v)==1 and int(v[0]["cycle_id"])==k;s[a]=v[0]
 x=float(s["O"]["update_l1"]);y=float(s["P"]["update_l1"]);assert x==y and s["O"]["input_shape"]==s["P"]["input_shape"];out.append(dict(sequence=n,cycle_id=k,position_group=t["position_group"],score=x,score_P=y,threshold=55.84033966064453,action="P" if math.isfinite(x) and x>55.84033966064453 else "O",nonfinite_fallback=not math.isfinite(x),score_frozen_before_label_join=True))
with (p/"MIDLATE_FROZEN_ACTIONS.csv").open("x",encoding="utf8",newline="") as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
print("prelabel_actions",len(out))
