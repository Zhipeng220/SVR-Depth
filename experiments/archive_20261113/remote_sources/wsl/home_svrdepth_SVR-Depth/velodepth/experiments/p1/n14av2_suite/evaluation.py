from .common import *
from .runtime import *
import gc,statistics,time

def npz(np,p,**arrays):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    with q.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(q,p);sync_directory(p.parent)

class Predictor:
    def __init__(self,ctx,method,seq,diagnostics=False,audit=False):
        self.ctx=ctx;self.method=method;self.seq=seq;self.eng=None;self.stream=None;self.micro=None;self.diagnostics=diagnostics;self.audit=audit
        t=ctx.rt.torch;self.t=t
        if method in ('C_star',*ctx.policy['arms']):
            self.eng=FrozenCarrier(ctx,seq)
            if method!='C_star':
                obj=t.load(ctx.out/'training'/method/'final.pt',map_location='cpu',weights_only=False)
                require(obj['signature']==ctx.signature and obj['arm']==method and obj['tensor_sha256']==ctx.rt.exp1.state_sha256(obj['state']),'Candidate final checkpoint identity differs')
                from .models import PhaseMicroReadout
                self.micro=PhaseMicroReadout(conditioned=method=='N14_A1').cuda();self.micro.load_state_dict(obj['state'],strict=True);self.micro.eval()
                for p in self.micro.parameters():p.requires_grad_(False)
            self.shape=self.eng.shape
        else:
            from y08y09_models import Stream,FastDepthAdapter
            self.model=ctx.rt.gt2.load_model(Path(ctx.args.snapshot),'off',t.device('cuda'));self.adapter=None
            if method=='O_fast':self.adapter=FastDepthAdapter().cuda();self.adapter.load_state_dict(ctx.ofast_state,strict=True)
            elif method=='O_multi':
                from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead
                self.adapter=PersistentFeatureRepairHead(channels=512,bottleneck=32,levels=4).cuda();self.adapter.load_state_dict(ctx.omulti_state,strict=True)
            for module in [self.model,self.adapter]:
                if module is not None:
                    module.eval()
                    for p in module.parameters():p.requires_grad_(False)
            self.shape=ctx.rt.exp1.shape_info_for(self.model,Path(ctx.manifests[seq][0]['rgb_path']),t.device('cuda'))
            cfg=dict(family=method,schedule='K'+str(ctx.policy['evaluation_K']),gain=1.,frequency=ctx.policy['baseline_output_frequency'])
            self.stream=Stream(ctx.rt,self.model,cfg,self.shape,self.adapter,debug=False)
    def reset(self):
        if self.eng is not None:self.eng.reset()
        if self.stream is not None:self.stream.reset()
    def step(self,i,record):
        with self.t.no_grad():
            if self.stream is not None:
                pred,meta=self.stream.step(i,record);meta.update(phase=phase(i),micro_applied=False,residual_norm=0.);return pred,meta
            frame=self.eng.load(i,False);native,static=self.eng.advance(frame,need_native_depth=self.micro is None or not micro_due(i) or self.audit);meta=dict(keyframe=i%32==0,age=i%32,phase=phase(i),micro_applied=self.micro is not None and micro_due(i))
            if self.audit:before=state_digest(self.eng.state);versions=state_versions(self.eng.state);original=self.eng.state
            if self.micro is None:p=native;reg=native.new_zeros(())
            else:p,reg=self.eng.micro(self.micro,native,static,i,self.diagnostics)
            if self.audit:
                require(self.eng.state is original and state_versions(self.eng.state)==versions,'Micro readout replaced/mutated persistent tensors')
                compare_state(before,state_digest(self.eng.state));meta['state_hashes']=before
                if phase(i)==0:require(self.t.equal(p,native),'h0 bypass changed C* output')
            out=p.detach().float().cpu().numpy();meta['residual_norm']=float(reg.item()) if self.diagnostics else 0.
            return out,meta
    def close(self):
        if self.eng is not None:self.eng.close();self.eng=None;self.micro=None
        if self.stream is not None:del self.stream,self.model,self.adapter;self.stream=None
        gc.collect();self.t.cuda.empty_cache()

def indices_for_probes(n):return sorted(set([0,min(8,n-1),min(16,n-1),n//2,n-1]))
def score_frame(ctx,s,i):return s not in ctx.old_sequences or i>=ctx.policy['tail_start'][s]

def quality(ctx,method,seq,save_depths=False,limit=None):
    rt=ctx.rt;records=ctx.manifests[seq][:limit];d=ctx.out/'quality'/method;p=d/(seq+'.csv');meta=p.with_suffix('.json')
    if meta.exists():
        m=read(meta);require(m['signature']==ctx.signature and m['csv_sha256']==sha(p),'Quality unit changed');verify_hashes(m['binary_sha256'])
        rr=csvread(p);require([int(r['index']) for r in rr]==list(range(len(records))),'Quality rows incomplete')
        return rr
    pred=Predictor(ctx,method,seq,diagnostics=True);rows=[];binaries={};probes=set(indices_for_probes(len(records)))
    try:
        for i,record in enumerate(records):
            value,info=pred.step(i,record);scored=(limit is not None or score_frame(ctx,seq,i)) and matched(record)
            gt=rt.exp1.load_frame(record,pred.shape,rt.torch.device('cuda'),need_gt=True)['gt_np'] if scored else None
            mm=metric(rt.np,value,gt)
            row=dict(method=method,sequence=seq,index=i,keyframe=info['keyframe'],age=info['age'],phase=phase(i),frame_type=frame_type(i),micro_applied=info['micro_applied'],residual_norm=info['residual_norm'],scored=scored,
                gt_match_status=record['gt_match_status'],prediction_sha256=hashlib.sha256(value.tobytes()).hexdigest(),**mm)
            rows.append(row)
            if i in probes or (save_depths and scored):
                path=ctx.out/'predictions'/method/seq/f'{i:06d}.npz';npz(rt.np,path,depth=value.astype('float32'));binaries[str(path)]=sha(path)
            if i==0 or (i+1)%200==0 or i==len(records)-1:print('[QUALITY]',method,seq,f'{i+1}/{len(records)}',flush=True)
        if method in ('A','C_star','O_fast','O_multi'):
            reference=HERE/'reference_metrics'/method/(seq+'.csv')
            if reference.exists():
                old={int(r['index']):r for r in csvread(reference)};errors=[]
                for r in rows:
                    ref=old[int(r['index'])]
                    if r['AbsRel']!='' and ref['AbsRel']!='':
                        require(int(r['valid_pixels'])==int(ref['valid_pixels']) and r['mask_sha256']==ref['mask_sha256'],'Baseline GT mask differs from reviewed N13')
                        errors.append(max(abs(float(r[k])-float(ref[k])) for k in ('AbsRel','RMSE','delta1')))
                require(not errors or max(errors)<=2e-5,'Frozen baseline quality differs from reviewed N13')
        csvwrite(p,rows);write(meta,dict(signature=ctx.signature,csv_sha256=sha(p),binary_sha256=binaries,RGB=len(records),scored=sum(r['scored'] for r in rows),probes=sorted(probes)))
        return rows
    finally:pred.close()

def timing(ctx,method,seq,repeat,limit=None):
    import gt4a2_y08cal_y09pareto as benchmark
    rt=ctx.rt;t=rt.torch;records=ctx.manifests[seq][:limit]
    p=ctx.out/'timing'/method/f'{seq}_r{repeat}.json'
    seal=p.with_suffix('.sha256.json')
    if p.exists() and seal.exists():
        s=read(seal);require(s['signature']==ctx.signature and s['sha256']==sha(p),'Timing file bytes changed')
        r=read(p);require(r['signature']==ctx.signature and r['verified'],'Unverified/foreign timing unit');return r
    pred=Predictor(ctx,method,seq);times=[];probes=set(indices_for_probes(len(records)))
    try:
        warm=min(ctx.policy['timing_warmup'],len(records))
        for i,record in enumerate(records[:warm]):pred.step(i,record)
        pred.reset()
        for r in records:Path(r['rgb_path']).read_bytes()
        t.cuda.synchronize();before=benchmark.gpu_endpoint(rt)
        require(before['clean'],'GPU endpoint is not clean; stop competing GPU process, then resume')
        for i,record in enumerate(records):
            start=time.perf_counter();value,_=pred.step(i,record);t.cuda.synchronize();times.append((time.perf_counter()-start)*1000.)
            if i in probes:
                with rt.np.load(ctx.out/'predictions'/method/seq/f'{i:06d}.npz') as z:reference=z['depth']
                require(rt.np.allclose(value,reference,rtol=1e-5,atol=1e-4),'Timing does not reproduce quality trajectory')
        after=benchmark.gpu_endpoint(rt)
        obj=dict(signature=ctx.signature,method=method,sequence=seq,repeat=repeat,RGB=len(times),mean_ms=statistics.mean(times),
            median_ms=statistics.median(times),p95_ms=float(rt.np.percentile(times,95)),before=before,after=after,verified=bool(before['clean'] and after['clean']),
            scope='warm-file-cache RGB read/decode/preprocess/DIS/native step/repair/readout/D2H/sync; model load, GT, metric, writes excluded',
            quality_probe_count=len(probes))
        if not obj['verified']:write(p.with_suffix('.rejected.json'),obj);raise RuntimeError('Timing contention; rejected trial retained, resume after other GPU work stops')
        write(p,obj);write(seal,dict(signature=ctx.signature,sha256=sha(p)));print('[TIMING]',method,seq,repeat,f'{obj["mean_ms"]:.3f} ms',flush=True);return obj
    finally:pred.close()

from .reporting import report

def evaluate_all(ctx,save_depths=False,smoke=False):
    methods=['A','O_fast','O_multi','C_star',*ctx.policy['arms']]
    seqs=[ctx.old_sequences[0]] if smoke else list(ctx.manifests)
    rows=[];times=[];limit=40 if smoke else None
    for s in seqs:
        for m in methods:rows.extend(quality(ctx,m,s,save_depths,limit))
    for r in range(1,2 if smoke else ctx.policy['timing_repeats']+1):
        for position,s in enumerate(seqs):
            rotated=methods[(r+position)%len(methods):]+methods[:(r+position)%len(methods)]
            for m in rotated:times.append(timing(ctx,m,s,r,limit))
    return report(ctx,rows,times,smoke)
