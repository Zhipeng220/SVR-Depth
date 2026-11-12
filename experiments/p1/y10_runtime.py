"""Audited fixed-K state interface. Torch is imported only by server preflight."""
from pathlib import Path
import gc, math, time
from y10_core import *

def flat_state(state):
    r,c,features,rays=state
    return {'radius':r,'confidence':c,'rays':rays,**{f'F{i}':f for i,f in enumerate(features)}}
def clone_state(state,device=None):
    def copy(t):
        t=t.detach().clone()
        return t.to(device) if device is not None else t
    r,c,f,rays=state
    return copy(r),copy(c),[copy(x) for x in f],copy(rays)
def to_cpu(state):return clone_state(state,'cpu')
def versions(state):return [(id(t),t._version) for t in flat_state(state).values()]
def finite_state(torch,state):return all(bool(torch.isfinite(t).all().item()) for t in flat_state(state).values())
def equal_state(torch,a,b,tol=0.):
    aa,bb=flat_state(a),flat_state(b)
    return set(aa)==set(bb) and all(aa[k].shape==bb[k].shape and bool(torch.allclose(aa[k],bb[k],rtol=0,atol=tol,equal_nan=False)) for k in aa)
def norm_delta(torch,a,b):
    numerator=sum(float((x.float()-y.float()).square().sum().item()) for x,y in zip(a,b))
    denominator=sum(float(y.float().square().sum().item()) for y in b)
    value=math.sqrt(numerator/max(denominator,1e-20))
    return value if math.isfinite(value) else None

def mix_components(native,repaired,commit_f,commit_rc):
    # Each component is explicitly committed at a *post-readout* boundary.
    # Crossed branches deliberately break the native radius=readout(F) relation.
    r,c,_,rays=repaired if commit_rc else native
    f=repaired[2] if commit_f else native[2]
    return r.detach().clone(),c.detach().clone(),[x.detach().clone() for x in f],native[3].detach().clone()

class Engine:
    def __init__(self,ctx,seq):
        self.ctx=ctx;self.rt=ctx.rt;self.seq=seq;torch=self.rt.torch
        self.model=self.rt.gt2.load_model(Path(ctx.args.snapshot).resolve(),'pre_readout',torch.device('cuda'))
        original=self.model.pixel_decoder.persistent_feature_repair
        original.load_state_dict(ctx.head_state,strict=True)
        require(sum(p.numel() for p in original.parameters())==170496,'Wrong head architecture')
        self.head_sha=self.rt.exp1.state_sha256(original.state_dict())
        class Controller(torch.nn.Module):
            def __init__(self,head):
                super().__init__();self.head=head;self.repeats=0;self.beta=1.;self.calls=0;self.last_trace=[]
            def forward(self,features):
                initial=features;self.last_trace=[]
                def record(j,updated):
                    self.calls+=1
                    self.last_trace.append(dict(iteration=j,beta=self.beta,relative_feature_change=norm_delta(torch,updated,initial)))
                return repeat_features(features,self.head,self.repeats,self.beta,record)
        self.control=Controller(original)
        self.model.pixel_decoder.persistent_feature_repair=self.control
        self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        records=ctx.manifests[seq]
        self.shape=self.rt.exp1.shape_info_for(self.model,Path(records[0]['rgb_path']),torch.device('cuda'))
        self.dis=self.rt.cv2.DISOpticalFlow_create(self.rt.cv2.DISOPTICAL_FLOW_PRESET_FAST)
        # Set the model's geometry caches before restoring an explicit state.
        frame=self.load(0,False)
        self.control.repeats=0
        self.rt.exp1.keyframe_state(self.model,frame,self.shape)
        require(tuple(self.model.pixel_decoder.radial_module.original_shapes)==tuple(self.shape['network_shape']),'Readout geometry differs')

    def close(self):
        require(self.rt.exp1.state_sha256(self.control.head.state_dict())==self.head_sha,'Frozen P weights were modified')
        del self.model,self.control;gc.collect();self.rt.torch.cuda.empty_cache()

    def load(self,index,gt=False):
        rec=self.ctx.manifests[self.seq][index]
        frame=self.rt.exp1.load_frame(rec,self.shape,self.rt.torch.device('cuda'),need_gt=gt)
        frame['sequence']=self.seq
        return frame

    def static(self,prev,frame,origin):
        with self.rt.torch.no_grad(),self.rt.torch.autocast('cuda',dtype=self.rt.torch.float16,enabled=origin=='P'):
            return self.rt.exp1.transition_static(self.model,prev,frame,self.shape,self.dis)

    def step(self,state,prev,frame,k,origin,repeats=0,beta=1.,static=None):
        torch=self.rt.torch;self.control.repeats=repeats;self.control.beta=beta
        i=int(frame['index']);before_calls=self.control.calls;self.control.last_trace=[]
        with torch.no_grad():
            if i%k==0:
                rays,r,c,f=self.rt.exp1.keyframe_state(self.model,frame,self.shape)
                result=(r,c,f,rays)
            else:
                require(state is not None and prev is not None,'Missing parent state/RGB')
                if static is None:static=self.static(prev,frame,origin)
                before=versions(state)
                with torch.autocast('cuda',dtype=torch.float16):
                    r,c,f=self.rt.gt2.fast_step(self.model,{'frames':[prev,frame],'fast_static':[static]},1,state[0],state[2])
                require(before==versions(state),'Native step mutated an input state in-place')
                result=(r,c,f,state[3])
            with torch.autocast('cuda',dtype=torch.float16):
                depth=self.rt.gt2.make_z_depth(result[0],result[3],self.shape['padded_shape'],self.shape['paddings'],self.model.interpolation_mode)
        calls=self.control.calls-before_calls
        require(calls==(0 if i%k==0 else repeats),'Repair head call budget differs')
        depth=depth.detach().float().cpu().numpy()
        require(tuple(depth.shape)==tuple(self.shape['original_shape']),'Restored depth shape differs')
        return result,depth,calls,list(self.control.last_trace)

    def metric(self,pred,frame):
        np=self.rt.np
        if frame['gt_np'] is None:return dict(matched=False,valid_prediction=bool(np.isfinite(pred).all() and (pred>0).all()),AbsRel='',RMSE='',delta1='',valid_pixels=0)
        gt=frame['gt_np'];mask=(gt>0)&np.isfinite(gt)
        require(mask.any(),'Empty GT valid mask')
        valid=bool(np.isfinite(pred[mask]).all() and (pred[mask]>0).all())
        if not valid:return dict(matched=True,valid_prediction=False,AbsRel='',RMSE='',delta1='',valid_pixels=int(mask.sum()))
        x,y=pred[mask].astype('float64'),gt[mask].astype('float64')
        return dict(matched=True,valid_prediction=True,AbsRel=float(np.mean(abs(x-y)/y)),
            RMSE=float(np.sqrt(np.mean((x-y)**2))),delta1=float(np.mean(np.maximum(x/y,y/x)<1.25)),valid_pixels=int(mask.sum()))

    def reference_audit(self,index,origin,k,metrics):
        ref=self.ctx.refs[(f'{origin}_K{k}_g{0 if origin=="A" else 1}_f1/raw',self.seq,index)]
        require(truth(ref['matched'])==metrics['matched'],'Reference GT identity differs')
        if metrics['matched'] and ref['AbsRel'] not in ('',None):
            require(metrics['valid_prediction'],'Natural carrier prediction invalid')
            error=max(abs(float(metrics[m])-float(ref[m])) for m in ('AbsRel','RMSE','delta1'))
            require(error<=2e-5,f'Natural {origin} K{k} differs from completed Y09 at {self.seq}/{index}: {error}')
            require(int(ref['valid_pixels'])==metrics['valid_pixels'],'GT pixel denominator differs')
            return error
        return None

def save_torch(ctx,path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    ctx.rt.torch.save(obj,tmp);tmp.replace(path)

def ensure_carrier(ctx,engine,k,origin,required):
    directory=ctx.out/'cache'/f'{engine.seq}_K{k}_{origin}'
    meta_path=directory/'complete.json';tensor_path=directory/'snapshots.pt'
    def publish_audit():
        dest=ctx.out/'natural_carriers'/f'{engine.seq}_K{k}_{origin}';dest.mkdir(parents=True,exist_ok=True)
        for name in ('complete.json','native_metrics.csv'):(dest/name).write_bytes((directory/name).read_bytes())
    if meta_path.exists():
        meta=read_json(meta_path)
        require(meta['signature']==ctx.signature and meta['snapshots_sha256']==digest(tensor_path),'Carrier cache changed')
        require(meta['required_snapshots']==sorted(required),'Carrier plan differs')
        require(meta['metrics_sha256']==digest(directory/'native_metrics.csv'),'Carrier audit CSV changed')
        publish_audit();return tensor_path
    torch=ctx.rt.torch;records=ctx.manifests[engine.seq]
    stop=max(required)+1 if ctx.args.smoke_only else len(records)
    state=prev=None;snapshots={};rows=[];errors=[];started=time.perf_counter()
    with torch.no_grad():
        for i in range(stop):
            matched=str(records[i].get('gt_match_status','')).strip().upper()=='MATCHED'
            # Match Y09's retained GT positions; all RGB still propagate.
            score=matched and (i<256 or i>=TAIL[engine.seq])
            frame=engine.load(i,score)
            state,depth,calls,_=engine.step(state,prev,frame,k,origin,repeats=1 if origin=='P' else 0)
            mm=engine.metric(depth,frame)
            if score:
                err=engine.reference_audit(i,origin,k,mm)
                if err is not None:errors.append(err)
                rows.append(dict(index=i,origin=origin,K=k,sequence=engine.seq,**mm,reference_max_error=err))
            if i in required:
                snapshots[i]=to_cpu(state)
            prev=frame
            if i==0 or (i+1)%200==0 or i+1==stop:print(f'[CARRIER] {engine.seq} K{k} {origin} {i+1}/{stop}',flush=True)
    require(set(snapshots)==set(required),'Carrier snapshots incomplete')
    write_csv(directory/'native_metrics.csv',rows)
    save_torch(ctx,tensor_path,dict(signature=ctx.signature,sequence=engine.seq,K=k,origin=origin,snapshots=snapshots))
    write_json(meta_path,dict(signature=ctx.signature,required_snapshots=sorted(required),processed_RGB=stop,
        snapshots_sha256=digest(tensor_path),metrics_sha256=digest(directory/'native_metrics.csv'),
        matched_reference_checks=len(errors),max_reference_metric_error=max(errors) if errors else None,
        diagnostic_wall_sec=time.perf_counter()-started))
    publish_audit();return tensor_path

def ensure_all_carriers(ctx,engine,case_list):
    paths={}
    for k in (16,32):
        for origin in ('A','P'):
            rr=[c['snapshot_index'] for c in case_list if c['sequence']==engine.seq and c['K']==k and c['origin']==origin]
            if rr:paths[k,origin]=ensure_carrier(ctx,engine,k,origin,set(rr))
    return paths

def snapshots_for(ctx,path):
    obj=ctx.rt.torch.load(path,map_location='cpu',weights_only=False)
    require(obj['signature']==ctx.signature,'Snapshot identity differs');return obj['snapshots']

def case_context(ctx,engine,case):
    records=ctx.manifests[case['sequence']];frames={};statics={}
    for h in range(-1,H+1):
        i=case['anchor']+h
        matched=h>=0 and str(records[i].get('gt_match_status','')).strip().upper()=='MATCHED'
        frames[h]=engine.load(i,matched)
        if h>=0 and i%case['K']!=0:statics[h]=engine.static(frames[h-1],frames[h],case['origin'])
    return frames,statics

def save_depth(ctx,folder,arm,h,pred,frame):
    if ctx.args.save_depth!='all' and h not in (0,4,8,16):return None
    np=ctx.rt.np;p=Path(folder)/'depth'/arm/f'h{h:02d}.npz';p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,depth=pred.astype('float32'),index=np.array(frame['index']),rgb_file=np.array(frame['rgb_file']))
    tmp.replace(p)
    if frame['gt_np'] is not None:
        gt_path=ctx.out/'gt'/f'{frame["sequence"]}_{frame["index"]:06d}.npz'
        if not gt_path.exists():
            gt_path.parent.mkdir(parents=True,exist_ok=True);t=gt_path.with_name(gt_path.name+'.tmp')
            with t.open('wb') as f:np.savez_compressed(f,gt=frame['gt_np'].astype('float32'),valid_mask=(frame['gt_np']>0)&np.isfinite(frame['gt_np']),gt_file=np.array(frame['gt_path'].name))
            t.replace(gt_path)
    return p

def row_for(ctx,case,arm,h,frame,metrics,calls,failed_reason='',**extra):
    return dict(case_id=case['case_id'],split=case['split'],sequence=case['sequence'],K=case['K'],origin=case['origin'],
        anchor=case['anchor'],age=case['age'],snapshot_index=case['snapshot_index'],arm=arm,h=h,index=frame['index'],
        after_reset=frame['index']>=case['next_keyframe'],keyframe=frame['index']%case['K']==0,
        rgb_file=frame['rgb_file'],rgb_timestamp=frame['rgb_timestamp'],gt_file=frame['gt_path'].name if frame.get('gt_path') else '',
        head_calls_cumulative=calls,failure_reason=failed_reason,**metrics,**extra)

def eval_branch(ctx,engine,case,folder,arm,start_state,start_pred,start_calls,frames,statics,schedule,natural_control=False):
    torch=ctx.rt.torch;state=clone_state(start_state,'cuda');rows=[];files={};calls=start_calls
    failed='';started=time.perf_counter()
    for h in range(H+1):
        frame=frames[h]
        if not failed:
            if h==0:pred=start_pred.copy()
            else:
                q,beta=schedule(h,frame['index'])
                state,pred,used,_=engine.step(state,frames[h-1],frame,case['K'],case['origin'],q,beta,statics.get(h));calls+=used
            valid_state=finite_state(torch,state);mm=engine.metric(pred,frame)
            if not valid_state or not mm['valid_prediction']:
                failed=f'nonfinite/nonpositive prediction or nonfinite state at h={h}'
                mm['valid_prediction']=False;mm.update(AbsRel='',RMSE='',delta1='')
            if not failed and natural_control:
                engine.reference_audit(frame['index'],case['origin'],case['K'],mm)
            p=save_depth(ctx,folder,arm,h,pred,frame)
            if p is not None:files[str(p.relative_to(folder))]=digest(p)
        else:
            mm=dict(matched=frame['gt_np'] is not None,valid_prediction=False,AbsRel='',RMSE='',delta1='',
                valid_pixels=int(((frame['gt_np']>0)&ctx.rt.np.isfinite(frame['gt_np'])).sum()) if frame['gt_np'] is not None else 0)
        rows.append(row_for(ctx,case,arm,h,frame,mm,calls,failed))
    return rows,files,dict(diagnostic_wall_sec=time.perf_counter()-started,head_calls=calls,failed=bool(failed),failure_reason=failed)

def commit_unit(ctx,folder,case,rows,files,invariants,extra=None):
    write_csv(Path(folder)/'frames.csv',rows)
    obj=dict(signature=ctx.signature,case=case,rows_sha256=digest(Path(folder)/'frames.csv'),audit_files=files,
        invariants=invariants,rows=len(rows),**(extra or {}))
    write_json(Path(folder)/'complete.json',obj)
    return obj

def prefix_metrics_natural(ctx,engine,case,arm,rows):
    # Only natural A/no-repair and natural P/continuous are compared to full replay.
    for r in rows:
        if r['arm']!=arm or not truth(r['matched']):continue
        engine.reference_audit(int(r['index']),case['origin'],case['K'],r)

def close_and_collect(ctx,phase,planned):
    records=[];metas=[]
    for case in planned:
        folder=ctx.out/phase/case['case_id'];meta=load_existing_unit(folder,ctx.signature)
        require(meta is not None,'Not every planned unit is complete: '+case['case_id'])
        records.extend(read_csv(folder/'frames.csv'));metas.append(meta)
    return records,metas

def execution_units(ctx,phase,planned,unit_runner,**kwargs):
    for seq in SEQS:
        selected=[c for c in planned if c['sequence']==seq]
        if not selected:continue
        pending=[c for c in selected if load_existing_unit(ctx.out/phase/c['case_id'],ctx.signature) is None]
        if not pending:
            print(f'[REUSE] {phase} {seq}: all {len(selected)} cases',flush=True);continue
        engine=Engine(ctx,seq)
        try:
            # Freeze the required cache plan, even when only a suffix remains unfinished.
            all_cases=ctx.plan['tail_cases']+ctx.plan.get('prefix_cases',[])
            carrier_paths=ensure_all_carriers(ctx,engine,all_cases)
            loaded={}
            for case in pending:
                key=(case['K'],case['origin'])
                if key not in loaded:loaded[key]=snapshots_for(ctx,carrier_paths[key])
                snapshot=loaded[key][case['snapshot_index']]
                unit_runner(ctx,engine,case,snapshot,ctx.out/phase/case['case_id'],**kwargs)
                print(f'[CASE COMPLETE] {phase} {case["case_id"]}',flush=True)
        finally:engine.close()
    return close_and_collect(ctx,phase,planned)
