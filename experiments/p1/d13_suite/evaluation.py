from .common import *
from .runtime import *
import gc,statistics,time

def npz(np,p,**arrays):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    with q.open('wb') as f:np.savez_compressed(f,**arrays)
    q.replace(p)

class Predictor:
    def __init__(self,ctx,method,seq):
        self.ctx=ctx;self.method=method;self.seq=seq;self.state=self.prev=None;self.eng=None;self.stream=None
        t=ctx.rt.torch;self.t=t
        if method in ('C_star',*ctx.policy['arms']):
            self.eng=AdaptedEngine(ctx,seq,None if method=='C_star' else method)
            if method!='C_star':
                obj=t.load(ctx.out/'training'/method/'final.pt',map_location='cpu',weights_only=False)
                require(obj['signature']==ctx.signature and obj['arm']==method and obj['tensor_sha256']==ctx.rt.exp1.state_sha256(obj['state']),'Candidate final checkpoint identity differs')
                self.eng.paths.load_state_dict(obj['state'],strict=True)
                for p in self.eng.paths.parameters():p.requires_grad_(False)
            self.shape=self.eng.base.shape
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
        self.state=self.prev=None
        if self.stream is not None:self.stream.reset()
    def step(self,i,record):
        with self.t.no_grad():
            if self.stream is not None:return self.stream.step(i,record)
            frame=self.eng.load(i,False);self.state,p,packet=self.eng.step(self.state,self.prev,frame)
            self.prev=frame;out=p.detach().float().cpu().numpy();self.eng.hook.packet=None
            return out,dict(keyframe=i%self.ctx.policy['evaluation_K']==0,age=i%self.ctx.policy['evaluation_K'])
    def close(self):
        if self.eng is not None:self.eng.close();self.eng=None
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
    pred=Predictor(ctx,method,seq);rows=[];binaries={};probes=set(indices_for_probes(len(records)))
    try:
        for i,record in enumerate(records):
            value,info=pred.step(i,record);scored=(limit is not None or score_frame(ctx,seq,i)) and matched(record)
            gt=rt.exp1.load_frame(record,pred.shape,rt.torch.device('cuda'),need_gt=True)['gt_np'] if scored else None
            mm=metric(rt.np,value,gt)
            row=dict(method=method,sequence=seq,index=i,keyframe=info['keyframe'],age=info['age'],scored=scored,
                gt_match_status=record['gt_match_status'],prediction_sha256=hashlib.sha256(value.tobytes()).hexdigest(),**mm)
            rows.append(row)
            if i in probes or (save_depths and scored):
                path=ctx.out/'predictions'/method/seq/f'{i:06d}.npz';npz(rt.np,path,depth=value.astype('float32'));binaries[str(path)]=sha(path)
            if i==0 or (i+1)%200==0 or i==len(records)-1:print('[QUALITY]',method,seq,f'{i+1}/{len(records)}',flush=True)
        csvwrite(p,rows);write(meta,dict(signature=ctx.signature,csv_sha256=sha(p),binary_sha256=binaries,RGB=len(records),scored=sum(r['scored'] for r in rows),probes=sorted(probes)))
        return rows
    finally:pred.close()

def timing(ctx,method,seq,repeat,limit=None):
    import gt4a2_y08cal_y09pareto as benchmark
    rt=ctx.rt;t=rt.torch;records=ctx.manifests[seq][:limit]
    p=ctx.out/'timing'/method/f'{seq}_r{repeat}.json'
    if p.exists():
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
        write(p,obj);print('[TIMING]',method,seq,repeat,f'{obj["mean_ms"]:.3f} ms',flush=True);return obj
    finally:pred.close()

def report(ctx,rows,times,smoke=False):
    np=ctx.rt.np;methods=['A','O_fast','O_multi','C_star',*ctx.policy['arms']]
    sequences=list(dict.fromkeys(r['sequence'] for r in rows));lookup={(r['method'],r['sequence'],int(r['index'])):r for r in rows}
    video=[]
    for m in methods:
        for s in sequences:
            rr=[r for r in rows if r['method']==m and r['sequence']==s and r['AbsRel']!='']
            require(bool(rr),'No scored frames in '+s)
            aa=[lookup['A',s,int(r['index'])] for r in rr]
            require(all(int(r['valid_pixels'])==int(a['valid_pixels']) and r['mask_sha256']==a['mask_sha256'] for r,a in zip(rr,aa)),'Method changed the GT mask/frame set')
            dif=[float(r['AbsRel'])-float(a['AbsRel']) for r,a in zip(rr,aa)]
            trials=[r for r in times if r['method']==m and r['sequence']==s]
            require(trials and all(r['verified'] for r in trials),'Timing incomplete/unverified')
            video.append(dict(method=m,sequence=s,frames=len(rr),AbsRel=statistics.mean(float(r['AbsRel']) for r in rr),
                RMSE=statistics.mean(float(r['RMSE']) for r in rr),delta1=statistics.mean(float(r['delta1']) for r in rr),
                p95_AbsRel=float(np.percentile([float(r['AbsRel']) for r in rr],95)),harmed_fraction=sum(d>0 for d in dif)/len(dif),
                severe_fraction=sum(d>.01 for d in dif)/len(dif),severe_frames=sum(d>.01 for d in dif),
                latency_ms=statistics.median(r['mean_ms'] for r in trials)))
    csvwrite(ctx.out/'per_recording.csv',video)
    macros=[]
    for group,seqs in [('old_tail',[s for s in sequences if s in ctx.old_sequences]),('Y11_development',[s for s in sequences if s not in ctx.old_sequences])]:
        if not seqs:continue
        for m in methods:
            vv=[r for r in video if r['method']==m and r['sequence'] in seqs]
            macros.append(dict(group=group,method=m,recordings=len(vv),**{k:statistics.mean(r[k] for r in vv) for k in ['AbsRel','RMSE','delta1','p95_AbsRel','harmed_fraction','severe_fraction','latency_ms']}))
    csvwrite(ctx.out/'macro_summary.csv',macros)
    gates={};threshold=ctx.policy['thresholds'];A,B=ctx.policy['arms']
    if not smoke:
        table={(r['group'],r['method']):r for r in macros};v={(r['method'],r['sequence']):r for r in video}
        for group in ('old_tail','Y11_development'):
            b=table[group,B];a=table[group,A];c=table[group,'C_star'];o=table[group,'O_multi']
            gates[group]=dict(gain_over_Cstar=1-b['AbsRel']/c['AbsRel'],beats_A=b['AbsRel']<a['AbsRel'],
                at_least_2pct_gain=b['AbsRel']<=c['AbsRel']*(1-threshold['min_gain_over_Cstar']),
                quality_ratio_to_Omulti=b['AbsRel']/o['AbsRel'],quality_gate=b['AbsRel']<=o['AbsRel']*threshold['quality_ratio_to_Omulti'],
                latency_ratio_to_Cstar=b['latency_ms']/c['latency_ms'],latency_gate=b['latency_ms']<=c['latency_ms']*threshold['max_latency_ratio_to_Cstar'])
        gates['balloon_severe_not_increased']=v[B,'balloon2']['severe_frames']<=v['C_star','balloon2']['severe_frames']
        gates['person_severe_excess']=v[B,'person_tracking2']['severe_fraction']-v['O_fast','person_tracking2']['severe_fraction']
        gates['person_severe_gate']=gates['person_severe_excess']<=threshold['max_person_severe_excess_over_Ofast']
        gates['per_recording_delta_B_minus_A']={s:v[B,s]['AbsRel']-v[A,s]['AbsRel'] for s in sequences}
    success=False
    if not smoke:
        success=all(gates[g][k] for g in ('old_tail','Y11_development') for k in ('beats_A','at_least_2pct_gain','quality_gate','latency_gate')) and gates['balloon_severe_not_increased'] and gates['person_severe_gate']
    summary=dict(signature=ctx.signature,experiment=ctx.policy['experiment'],status='SMOKE_COMPLETE' if smoke else 'DEVELOPMENT_SUITE_COMPLETE',
        training_seeds=1,quality_units=len(methods)*len(sequences),timing_trials=len(times),all_timing_verified=all(r['verified'] for r in times),
        screening_gate_passed=bool(success),method_pass=False,Y11_pass=False,final_blind_test=False,gate_details=gates,
        comparison=ctx.policy['attribution'],compute_claim=ctx.policy['compute_claim'],
        prediction_retention=ctx.plan['prediction_retention'],
        deployment='K32; C* and D/N residuals on S8, frozen output baselines on every non-keyframe. These are new development conditions, not the old Y11 K16/f1 certification.')
    write(ctx.out/'summary.json',summary)
    lines=[f'# {ctx.policy["experiment"]} {ctx.policy["method"]} 开发实验报告','',
        '状态：'+summary['status'],'',f'筛查门槛：{success}；完整方法通过：False；最终盲测：False。','',
        '固定 C*，训练 A/B 两组；仅使用旧录像前缀。旧尾段与已查看的 Y11 五组录像均属于开发数据。',
        '同更新次数及目标帧曝光，不宣称相等训练计算量。实际费用见 training_summary.json。','',
        '|分组|方法|宏 AbsRel|宏 RMSE|δ1|严重伤害率|延迟 ms|','|---|---|---:|---:|---:|---:|---:|']
    for r in macros:lines.append(f'|{r["group"]}|{r["method"]}|{r["AbsRel"]:.6f}|{r["RMSE"]:.6f}|{r["delta1"]:.6f}|{r["severe_fraction"]:.4%}|{r["latency_ms"]:.3f}|')
    lines+=['','每个录像等权；伤害以相同 K32 原生 A 为参照，严重伤害为逐帧 AbsRel 增量 >0.01。',
        '功能保持只约束当前读出变化，不保证未来稳定。D 的包络只约束块 RMS，风险标签为训练期帧级 H1 反事实，均不构成理论安全认证。',
        'A/B 的所有逐录像差值与新冻结门槛见 summary.json。单种子筛查通过也不能自动进入最终主实验。']
    (ctx.out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');return summary

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
