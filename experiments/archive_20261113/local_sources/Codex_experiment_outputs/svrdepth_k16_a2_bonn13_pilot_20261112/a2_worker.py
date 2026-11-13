import os,sys,csv,json,time,hashlib,traceback
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="0";os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np,torch,cv2
ROOT=Path("/home/svrdepth/SVR-Depth/velodepth");sys.path[:0]=[str(ROOT),str(ROOT/"experiments/p1")]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1,cvpr_m04_closed_loop as m04
name,arm=sys.argv[1:];assert arm in ["O","P"]
out=ROOT/"outputs/svrdepth_k16_a2_bonn13_pilot_20261112";dest=out/"trajectories"/(name+"_"+arm);dest.mkdir(parents=True,exist_ok=False);start=time.monotonic();counts={"frames_completed":0,"keyframe":0,"fast_off":0,"fast_on":0,"repair_hook":0};cur=-1; scores=[];rows=[];status={"sequence":name,"arm":arm,"status":"FAIL"}
try:
 snap=Path.home()/"hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c";v5=ROOT/"outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt";model=gt2.load_model(snap,"pre_readout",torch.device("cuda:0"));model.pixel_decoder.persistent_feature_repair.load_state_dict(torch.load(v5,map_location="cpu",weights_only=False)["repair_head_state_dict"],strict=True);model.eval();device=torch.device("cuda:0");torch.cuda.reset_peak_memory_stats()
 records=exp1.read_manifest(ROOT/"outputs/y04_v4_manifest_candidate_20261108_190940"/(name+"_pairs_v4_candidate.csv"),training=False)[:64];assert len(records)==64;shape=exp1.shape_info_for(model,Path(records[0]["rgb_path"]),device);assert tuple(shape["original_shape"])==(480,640) and tuple(shape["network_shape"])==(434,574);dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);capt=[]
 def hook(m,inp,outp):
  counts["repair_hook"]+=1
  if cur%16==1:
   aa=inp[0];bb=outp;assert len(aa)==len(bb)==4;u=[float(torch.linalg.vector_norm((y.float()-x.float()).reshape(-1))) for x,y in zip(aa,bb)];capt.append(dict(sequence=name,arm=arm,cycle_id=cur//16,frame=cur,update_l1=u[0],update_overall=sum(x*x for x in u)**.5,input_shape=str(tuple(aa[0].shape)),output_shape=str(tuple(bb[0].shape)),num_elements=aa[0].numel(),dtype=str(aa[0].dtype),repair_mode=str(model.pixel_decoder.feature_repair_mode),score_provenance="Repair head ON input/output same call FP32 subtract"))
 handle=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook);prev=None
 with (dest/"frames.csv").open("x",newline="") as f:
  fields=["sequence","arm","frame","cycle_id","age","status","valid_pixels","mask_sha256","AbsRel","RMSE","delta1"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();f.flush()
  with torch.no_grad():
   for i,rec in enumerate(records):
    cur=i;match=str(rec.get("gt_match_status","")).upper()=="MATCHED";frame=exp1.load_frame(rec,shape,device,need_gt=match)
    if i%16==0:
     counts["keyframe"]+=1;model.pixel_decoder.feature_repair_mode="off";rays,r,c,features=exp1.keyframe_state(model,frame,shape);state={"radius":r,"confidence":c,"features":features,"rays_map":rays};pred=gt2.make_z_depth(r,rays,shape["padded_shape"],shape["paddings"],model.interpolation_mode).detach().float().cpu().numpy()
    else:
     static=exp1.transition_static(model,prev,frame,shape,dis);state,pred,steps=m04.apply_branch(model,exp1,gt2,prev,frame,static,state,"O_same_head" if arm=="O" else "P_full",shape,torch);counts["fast_off"]+=int(arm=="O");counts["fast_on"]+=1
    row=dict(sequence=name,arm=arm,frame=i,cycle_id=i//16,age=i%16,status="NO_GT",valid_pixels="",mask_sha256="",AbsRel="",RMSE="",delta1="")
    if match:
     gt=frame["gt_np"];mask=(gt>0)&np.isfinite(gt)
     if not mask.any():row["status"]="NO_VALID_GT"
     elif not np.all(np.isfinite(pred[mask])&(pred[mask]>0)):row["status"]="INVALID_PRED"
     else:
      g=gt[mask].astype(np.float64);pr=pred[mask].astype(np.float64);row.update(status="OK",valid_pixels=int(mask.sum()),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),AbsRel=float(np.mean(abs(g-pr)/g)),RMSE=float(np.sqrt(np.mean((g-pr)**2))),delta1=float(np.mean(np.maximum(g/pr,pr/g)<1.25)))
    w.writerow(row);f.flush();counts["frames_completed"]+=1;prev=frame
    with (out/"A2_RUNTIME_LEDGER.jsonl").open("a") as led:led.write(json.dumps(dict(sequence=name,arm=arm,frame=i,counts=counts))+"\n")
 handle.remove();assert len(capt)==4 and all(x["repair_mode"]=="pre_readout" for x in capt);scores=capt;status.update(status="PASS",elapsed_seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,counts=counts)
except Exception:status["error"]=traceback.format_exc();print(status["error"],flush=True)
finally:
 if scores:
  with (dest/"scores.csv").open("x",newline="") as f:w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
 (dest/"status.json").write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
if status["status"]!="PASS":sys.exit(1)
