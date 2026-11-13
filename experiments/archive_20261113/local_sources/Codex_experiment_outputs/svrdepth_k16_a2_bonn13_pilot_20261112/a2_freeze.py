from pathlib import Path
import json,csv,datetime
r=Path("/home/svrdepth/SVR-Depth/velodepth");p=r/"outputs/svrdepth_k16_a2_bonn13_pilot_20261112";p.mkdir(exist_ok=False);d=json.loads((r/"outputs/y04_v5_preserve_seed20261025_20261109_112607/run_identity.json").read_text());seq=d["training_sequences"];assert len(seq)==13 and len(set(seq))==13
se=[];pilot=[];reserve=[]
for name in seq:
 path=r/"outputs/y04_v4_manifest_candidate_20261108_190940"/(name+"_pairs_v4_candidate.csv")
 with path.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
 idx=all(int(x["frame_index_zero_based"])==i for i,x in enumerate(rows));ts=all(float(rows[i]["rgb_timestamp"])>float(rows[i-1]["rgb_timestamp"]) for i in range(1,min(64,len(rows))));present=sum(Path(x["rgb_path"]).is_file() for x in rows[:64]);gt=sum(x["gt_match_status"].upper()=="MATCHED" and Path(x["depth_path"]).is_file() for x in rows[:64]);full=len(rows)//16;eligible=len(rows)>=64 and idx and ts and present==64
 se.append(dict(sequence=name,manifest=str(path),frames=len(rows),first64_rgb_existing=present,first64_matched_gt_existing=gt,indices_contiguous=idx,timestamps_increasing_first64=ts,physical_scene_id="UNKNOWN",training_exposure="V5_REPAIR_TRAINING",resolution="VERIFY_AT_RUNTIME",depth_scale="RAW_BONN_LOADER_5000",eligible=eligible))
 for j in range(4):pilot.append(dict(sequence=name,cycle_id=j,start_frame=16*j,end_frame=16*j+15,decision_frame=16*j+1,eligible=eligible))
 candidates=sorted(set([full//2,full-1])-set(range(4)))
 for j in candidates:
  if j<full:reserve.append(dict(sequence=name,cycle_id=j,start_frame=16*j,end_frame=16*j+15,reason="middle" if j==full//2 else "last_complete"))
def save(n,x):
 with (p/n).open("w",encoding="utf8",newline="") as f:w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
save("BONN13_SEQUENCE_MANIFEST.csv",se);save("BONN13_PILOT_CYCLE_MANIFEST.csv",pilot);save("BONN13_MIDLATE_RESERVE_MANIFEST.csv",reserve)
(p/"A2_PROTOCOL_FREEZE.json").write_text(json.dumps({"timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"sequence_order":seq,"first_cycles":[0,1,2,3],"reserve_rule":"floor(full_cycles/2), full_cycles-1 excluding first four", "score":"FP32 repair output minus repair input first layer vector L2","checkpoint_sha256":"2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251","threshold":"not fitted","prior_to_gpu":True},indent=2))
print("eligible",sum(x["eligible"] for x in se),"of",len(se));print([(x["sequence"],x["frames"],x["first64_matched_gt_existing"]) for x in se])
