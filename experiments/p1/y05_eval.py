"""Full RGB deployment, paired single-pulse probes, and separate prefix timing."""
from y05_common import *
from y05_engine import Engine,ensure_bank,parent_for,detached_state,numpy_metrics,valid_state
from y05_train import head_states
import statistics,time

def spec(head,k=32,kind='S8',gain=1.,output=False):
    return dict(id=f'{head}_K{k}_{kind}_g{gain:g}'.replace('.','p'),head=head,K=k,kind=kind,gain=gain,output=output)
def configurations():
    result=[spec('A',k,'none',0.) for k in (16,32)]
    result += [spec('P_init',k) for k in (16,32)]
    result += [spec('P_init',32,gain=g) for g in (.25,.5)]
    result += [spec(h) for h in ('C_current','F_future','FH_future_harm','P_equal_updates','P_equal_compute')]
    result += [spec('FH_future_harm',16)]
    result += [spec(h,k,'continuous') for h in ('FH_future_harm','P_init') for k in (16,32)]
    result += [spec('O_fast',32,'continuous',output=True)]
    require(len({s['id'] for s in result})==17,'Configuration grid changed')
    return result

def prepare_spec(ctx,engine,cfg,states):
    engine.head.load_state_dict(states.get(cfg['head'],ctx.head_state),strict=True)
    adapter=None
    if cfg['output']:
        adapter=ctx.oldmodels.FastDepthAdapter().cuda();adapter.load_state_dict(ctx.ofast_state,strict=True);adapter.eval()
        for p in adapter.parameters():p.requires_grad_(False)
    return adapter

def advance(ctx,engine,cfg,adapter,state,previous,frame,audit=True):
    torch=ctx.rt.torch;i=frame['index'];is_key=i%cfg['K']==0
    apply=cfg['head'] not in ('A','O_fast') and arm_active(cfg['kind'],i,cfg['K'])
    origin='A' if cfg['head'] in ('A','O_fast') else 'S8'
    with torch.no_grad():
        state,_=engine.tensor_step(state,previous,frame,cfg['K'],origin,apply,gain=cfg['gain'])
        if audit:require(valid_state(torch,state),'Nonfinite deployed state')
        with torch.autocast('cuda',dtype=torch.float16):
            pred=engine.read_depth(state)
            if adapter is not None and not is_key:
                before=[(id(x),x._version) for x in (state[0],state[1],*state[2])] if audit else None
                pred=adapter(pred,ctx.oldmodels.rgb_for_output(frame,engine.shape),cfg['gain'])
                if audit:require(before==[(id(x),x._version) for x in (state[0],state[1],*state[2])],'Output adapter changed recursive state')
        pred=pred.detach().float().cpu().numpy()
    return state,pred,dict(keyframe=is_key,age=i%cfg['K'],repair_applied=apply,head_calls=int(apply),output_adapter_calls=int(adapter is not None and not is_key))

def historical(ctx,cfg,seq,index,metrics):
    variant=None
    if cfg['head']=='A':variant=f'A_K{cfg["K"]}_g0_f1/raw'
    elif cfg['head']=='P_init' and cfg['kind']=='continuous':variant=f'P_K{cfg["K"]}_g1_f1/raw'
    elif cfg['head']=='O_fast':variant='O_fast_K32_g1_f1/raw'
    if variant and metrics['AbsRel']!='':
        ref=ctx.refs[variant,seq,index]
        error=max(abs(float(ref[m])-metrics[m]) for m in ('AbsRel','RMSE','delta1'))
        require(error<=2e-5 and int(ref['valid_pixels'])==metrics['valid_pixels'],'Existing control differs from Y09: '+variant)
        return error
    return None

def quality_unit(ctx,engine,cfg,seq,states):
    folder=ctx.out/'quality'/cfg['id'];csv_path=folder/(seq+'.csv');meta_path=folder/(seq+'.json')
    if meta_path.exists():
        m=jread(meta_path);require(m['signature']==ctx.signature and m['config']==cfg and m['csv_sha256']==sha(csv_path),'Quality identity changed');return
    engine.configure(seq);adapter=prepare_spec(ctx,engine,cfg,states);rows=[];state=previous=None;errors=[]
    stop=32 if ctx.args.smoke_only else len(ctx.manifests[seq]);start=time.perf_counter()
    for i in range(stop):
        keep=i<256 or i>=TAIL[seq];frame=engine.load(i,keep)
        state,pred,counts=advance(ctx,engine,cfg,adapter,state,previous,frame)
        metrics=numpy_metrics(ctx.rt.np,pred,frame);err=historical(ctx,cfg,seq,i,metrics)
        if err is not None:errors.append(err)
        endpoints=i in (0,7,8,31,32,255,TAIL[seq],stop-1)
        row=dict(config_id=cfg['id'],sequence=seq,index=i,matched=matched(ctx.manifests[seq][i]),scored=ctx.args.smoke_only or i>=TAIL[seq],metric_retained=keep,
            rgb_file=frame['rgb_file'],gt_file=frame['gt_path'].name if frame['gt_path'] else '',**counts,**metrics,
            prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest() if endpoints else '')
        rows.append(row);previous=frame
        if i==0 or (i+1)%200==0 or i==stop-1:print(f'[QUALITY] {cfg["id"]}/{seq} {i+1}/{stop}',flush=True)
    cwrite(csv_path,rows);jwrite(meta_path,dict(signature=ctx.signature,config=cfg,csv_sha256=sha(csv_path),RGB_frames=stop,
        historical_reference_checks=len(errors),max_historical_metric_error=max(errors) if errors else None,diagnostic_wall_sec=time.perf_counter()-start,
        checkpoint='fixed last update; no tail selection'))
    del adapter

def run_quality(ctx):
    states,paths=head_states(ctx);jwrite(ctx.out/'selected_checkpoints.json',paths);engine=Engine(ctx,SEQS[0])
    try:
        for cfg in configurations():
            for seq in SEQS:quality_unit(ctx,engine,cfg,seq,states)
    finally:engine.close()

def save_probe(ctx,folder,arm,h,pred,frame):
    if h not in (0,4,8,16):return
    path=folder/'depth'/arm/f'h{h:02d}.npz';path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    obj=dict(depth=pred.astype('float32'),index=ctx.rt.np.array(frame['index']))
    if frame['gt_np'] is not None:obj.update(gt=frame['gt_np'],valid_mask=(frame['gt_np']>0)&ctx.rt.np.isfinite(frame['gt_np']))
    with tmp.open('wb') as f:ctx.rt.np.savez_compressed(f,**obj)
    tmp.replace(path)

def run_probes(ctx):
    states,_=head_states(ctx);engine=Engine(ctx,SEQS[0]);cases=ctx.plan['probe_plan'];paths=ensure_bank(ctx,engine,cases,'probe')
    names=['no_pulse','P_init','C_current','F_future','FH_future_harm','P_equal_updates','P_equal_compute']
    try:
        for case in cases:
            folder=ctx.out/'probes'/case['case_id'];meta=folder/'complete.json';csv=folder/'frames.csv'
            if meta.exists():
                m=jread(meta);require(m['signature']==ctx.signature and m['rows_sha256']==sha(csv),'Probe identity changed')
                for name,digest in m['saved_array_sha256'].items():require((folder/name).is_file() and sha(folder/name)==digest,'Saved probe audit missing/changed: '+name)
                continue
            frames,statics=engine.context(case);parent=parent_for(ctx,paths,case);rows=[];depth_hashes={}
            for name in names:
                engine.head.load_state_dict(states.get(name,ctx.head_state),strict=True);state=detached_state(parent,'cuda')
                for h in range(17):
                    state,pred=engine.eval_step(state,frames[h-1],frames[h],case['K'],case['origin'],apply=h==0 and name!='no_pulse',static=statics.get(h))
                    mm=numpy_metrics(ctx.rt.np,pred,frames[h]);save_probe(ctx,folder,name,h,pred,frames[h])
                    if h in (0,4,8,16):depth_hashes[f'depth/{name}/h{h:02d}.npz']=sha(folder/'depth'/name/f'h{h:02d}.npz')
                    rows.append(dict(**{k:case[k] for k in ('case_id','sequence','origin','K','anchor','age','split')},arm=name,h=h,index=case['anchor']+h,matched=frames[h]['gt_np'] is not None,
                        after_reset=case['anchor']+h>=(case['anchor']//32+1)*32,head_calls_cumulative=int(name!='no_pulse'),**mm))
                del state
            require(len(rows)==7*17,'Probe trajectory incomplete')
            control={(r['h']):r for r in rows if r['arm']=='no_pulse'}
            for r in rows:
                if r['after_reset'] and r['matched']:
                    require(max(abs(float(r[m])-float(control[r['h']][m])) for m in ('AbsRel','RMSE','delta1'))<=1e-7,'Pulse probe survived full reset')
            cwrite(csv,rows);jwrite(meta,dict(signature=ctx.signature,case=case,rows_sha256=sha(csv),rows=len(rows),saved_array_sha256=depth_hashes,future_repair_disabled=True))
            print('[PROBE COMPLETE]',case['case_id'],flush=True)
            del frames,statics,parent
    finally:engine.close()

def probe_windows(ctx):
    output=[]
    for case in ctx.plan['probe_plan']:
        folder=ctx.out/'probes'/case['case_id'];meta=jread(folder/'complete.json');require(meta['signature']==ctx.signature and sha(folder/'frames.csv')==meta['rows_sha256'],'Probe changed')
        rows=cread(folder/'frames.csv');by={}
        for r in rows:by.setdefault(r['arm'],{})[int(r['h'])]=r
        require(all(set(x)==set(range(17)) for x in by.values()),'Missing planned probe RGB')
        for arm,rr in by.items():
            for horizon in (4,8,16):
                shared=[h for h in range(1,horizon+1) if truth(by['no_pulse'][h]['matched'])]
                require(all(truth(rr[h]['matched'])==truth(by['no_pulse'][h]['matched']) for h in range(17)),'Probe GT pairing changed')
                a=[float(rr[h]['AbsRel']) for h in shared];b=[float(by['no_pulse'][h]['AbsRel']) for h in shared];d=[x-y for x,y in zip(a,b)]
                before=[h for h in shared if not truth(rr[h]['after_reset'])]
                output.append(dict(**{k:case[k] for k in ('case_id','sequence','origin','K','anchor','age')},arm=arm,horizon=horizon,status='OK' if a else 'NO_GT',n_GT=len(a),
                    mean_AbsRel=statistics.mean(a) if a else '',cumulative_gain=sum(y-x for x,y in zip(a,b)) if a else '',
                    harmed_fraction=sum(x>1e-8 for x in d)/len(d) if d else '',harm_gt_0p01=sum(x>.01 for x in d),
                    pre_reset_gain=sum(float(by['no_pulse'][h]['AbsRel'])-float(rr[h]['AbsRel']) for h in before),n_pre_reset_GT=len(before),current_excluded=True))
    return output

def timing_unit(ctx,engine,cfg,seq,repeat,states):
    folder=ctx.out/'timing';path=folder/f'r{repeat}_{cfg["id"]}_{seq}.json'
    if path.exists():
        m=jread(path);require(m['signature']==ctx.signature and m['config']==cfg and m['endpoint_verified'],'Timing identity changed');return
    engine.configure(seq);adapter=prepare_spec(ctx,engine,cfg,states);quality={int(r['index']):r for r in cread(ctx.out/'quality'/cfg['id']/(seq+'.csv'))}
    stop=ctx.plan['settings']['timing_RGB'];warmup=ctx.plan['settings']['timing_warmup'];state=previous=None;elapsed=[];counts=dict(keyframes=0,head_calls=0,output_adapter_calls=0);verified=[]
    for i in range(stop):
        frame=engine.load(i,False);ctx.rt.torch.cuda.synchronize();start=time.perf_counter()
        state,pred,meta=advance(ctx,engine,cfg,adapter,state,previous,frame,audit=False)
        ctx.rt.torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000.
        require(bool(ctx.rt.np.isfinite(pred).all() and (pred>0).all()),'Invalid timing prediction')
        if quality[i]['prediction_sha256']:
            require(hashlib.sha256(pred.tobytes()).hexdigest()==quality[i]['prediction_sha256'],'Timing endpoint differs from quality')
            verified.append(i)
        if i>=warmup:
            elapsed.append(ms);counts['keyframes']+=int(meta['keyframe']);counts['head_calls']+=meta['head_calls'];counts['output_adapter_calls']+=meta['output_adapter_calls']
        previous=frame
    require(len(elapsed)==stop-warmup and stop-1 in verified,'Timing endpoint/sample missing')
    jwrite(path,dict(signature=ctx.signature,config=cfg,sequence=seq,repeat=repeat,warmup_RGB=warmup,n_RGB=len(elapsed),frame_ms=elapsed,counts=counts,
        mean_ms=statistics.mean(elapsed),p50_ms=float(ctx.rt.np.quantile(elapsed,.5)),p95_ms=float(ctx.rt.np.quantile(elapsed,.95)),endpoint_verified=True,verified_indices=verified,scope=ctx.plan['settings']['timing_scope']))
    print(f'[TIMING] r{repeat} {cfg["id"]}/{seq} {statistics.mean(elapsed):.3f}ms',flush=True);del adapter

def run_timing(ctx):
    states,_=head_states(ctx);configs=[s for s in configurations() if s['K']==32];engine=Engine(ctx,SEQS[0])
    try:
        for repeat in range(ctx.plan['settings']['timing_repeats']):
            rotated=configs[repeat:]+configs[:repeat]
            for cfg in rotated:
                for seq in SEQS:timing_unit(ctx,engine,cfg,seq,repeat+1,states)
    finally:engine.close()

def report(ctx):
    for label,checkpoint in jread(ctx.out/'selected_checkpoints.json').items():
        require(sha(checkpoint['path'])==checkpoint['sha256'],'Evaluated checkpoint changed before report: '+label)
    all_rows=[];per_sequence=[];macro=[]
    for cfg in configurations():
        for seq in SEQS:
            path=ctx.out/'quality'/cfg['id']/(seq+'.csv');m=jread(path.with_suffix('.json'))
            require(m['signature']==ctx.signature and sha(path)==m['csv_sha256'],'Quality changed before reporting')
            rows=cread(path);require(len(rows)==(32 if ctx.args.smoke_only else N_RGB[seq]),'Incomplete quality RGB')
            all_rows.extend(rows);scored=[r for r in rows if truth(r['scored']) and truth(r['matched'])]
            require(scored and all(r['AbsRel']!='' for r in scored),'Missing scored prediction')
            base=cread(ctx.out/'quality'/spec('A',cfg['K'],'none',0.)['id']/(seq+'.csv'));a={int(r['index']):r for r in base}
            delta=[float(r['AbsRel'])-float(a[int(r['index'])]['AbsRel']) for r in scored]
            native=statistics.mean(float(a[int(r['index'])]['AbsRel']) for r in scored);mean=statistics.mean(float(r['AbsRel']) for r in scored)
            per_sequence.append(dict(config_id=cfg['id'],sequence=seq,K=cfg['K'],n_GT=len(scored),mean_AbsRel=mean,mean_RMSE=statistics.mean(float(r['RMSE']) for r in scored),mean_delta1=statistics.mean(float(r['delta1']) for r in scored),A_AbsRel=native,relative_AbsRel_change=mean/native-1,
                harmed_fraction=sum(x>1e-8 for x in delta)/len(delta),harm_gt_0p01=sum(x>.01 for x in delta),harm_gt_0p02=sum(x>.02 for x in delta),frame_AbsRel_p95=float(ctx.rt.np.quantile([float(r['AbsRel']) for r in scored],.95))))
        rr=[r for r in per_sequence if r['config_id']==cfg['id']]
        macro.append(dict(config_id=cfg['id'],K=cfg['K'],macro_AbsRel=statistics.mean(r['mean_AbsRel'] for r in rr),macro_RMSE=statistics.mean(r['mean_RMSE'] for r in rr),macro_delta1=statistics.mean(r['mean_delta1'] for r in rr),macro_harmed_fraction=statistics.mean(r['harmed_fraction'] for r in rr),worst_video_AbsRel=max(r['mean_AbsRel'] for r in rr),worst_video_relative_change=max(r['relative_AbsRel_change'] for r in rr),mean_sequence_p95=statistics.mean(r['frame_AbsRel_p95'] for r in rr)))
    cwrite(ctx.out/'y05_frame_metrics.csv',all_rows);cwrite(ctx.out/'y05_sequence_quality.csv',per_sequence);cwrite(ctx.out/'y05_macro_quality.csv',macro)
    windows=probe_windows(ctx);cwrite(ctx.out/'y05_paired_future_windows.csv',windows)
    groups={}
    for r in windows:groups.setdefault((r['sequence'],r['origin'],r['arm'],r['horizon']),[]).append(r)
    grouped=[]
    for key,rr in sorted(groups.items()):
        ok=all(r['status']=='OK' for r in rr)
        grouped.append(dict(zip(('sequence','origin','arm','horizon'),key),n_cases=len(rr),status='OK' if ok else 'NO_GT',**{m:statistics.mean(float(r[m]) for r in rr) if ok else '' for m in ('mean_AbsRel','cumulative_gain','harmed_fraction','pre_reset_gain')}))
    cwrite(ctx.out/'y05_probe_sequence_summary.csv',grouped)
    timing=[]
    configs=[s for s in configurations() if s['K']==32]
    for cfg in configs:
        for seq in SEQS:
            trials=[jread(ctx.out/'timing'/f'r{r}_{cfg["id"]}_{seq}.json') for r in range(1,ctx.plan['settings']['timing_repeats']+1)]
            require(all(t['signature']==ctx.signature and t['endpoint_verified'] for t in trials),'Timing incomplete')
            timing.append(dict(config_id=cfg['id'],sequence=seq,repeats=len(trials),mean_ms=statistics.mean(t['mean_ms'] for t in trials),mean_p50_ms=statistics.mean(t['p50_ms'] for t in trials),mean_p95_ms=statistics.mean(t['p95_ms'] for t in trials),mean_ms_min=min(t['mean_ms'] for t in trials),mean_ms_max=max(t['mean_ms'] for t in trials),n_RGB_per_repeat=trials[0]['n_RGB'],scope=ctx.plan['settings']['timing_scope']))
    cwrite(ctx.out/'y05_timing_summary.csv',timing)
    train=jread(ctx.out/'training_summary.json');gradient=jread(ctx.out/'gradient_audit.json');require(gradient['passed'] and gradient['signature']==ctx.signature,'Gradient mechanics missing')
    for k,v in train.items():require(v['signature']==ctx.signature,'Training identity changed: '+k)
    summary=dict(experiment='Y05',version=VERSION,signature=ctx.signature,status='SMOKE_PASS' if ctx.args.smoke_only else 'Y05_DEVELOPMENT_SUITE_COMPLETE',partial_smoke=ctx.args.smoke_only,
        training_performed=True,independent_testing_performed=False,Y11_pass=False,no_automatic_method_pass=True,
        horizon=4,trained_arms=list(TRAIN_ARMS),quality_configurations=len(configurations()),quality_RGB_records=len(all_rows),probe_cases=len(ctx.plan['probe_plan']),probe_frame_records=len(ctx.plan['probe_plan'])*7*17,
        timing_trials=len(configs)*3*ctx.plan['settings']['timing_repeats'],gradient_audit_passed=True,checkpoint_selection='fixed final update; no tail/gain/scene selection',numerical_failure_branches=0,
        interpretation='Read per-video risk, single-pulse future net gain, current-only/future-only/closed-loop controls and measured training costs; completion is not a method PASS',
        costs='timing is prefix benchmark; compute control matches committed CUDA full-step event interval within one step, not FLOPs/exposures/total physical wall; cache, I/O checkpoints and repeats reported separately')
    jwrite(ctx.out/'summary.json',summary)
    (ctx.out/'RESULT_README.md').write_text('''# Y05-H4 结果阅读

先看 gradient_audit.json，再看 training_summary.json 的实际预算与 inherited P 暴露。
主配置 FH_future_harm_K32_S8_g1；不按尾段挑 checkpoint/gain/视频。
对照 C_current、F_future、P_equal_updates、P_equal_compute、旧 P gain0.25/0.5/1 和 O_fast。
先看 y05_sequence_quality.csv 的 balloon2 与最差视频，再看宏平均和 y05_paired_future_windows.csv。
probe 的 A/S8 是冻结初始 P 的共同离策略父状态；不是各新模型自己的在线历史。
稀疏部署从 RGB0 连续运行；连续 P 是压力测试，不是训练调度；无 Auto 稳定性主张。
timing 使用三视频前256RGB、前32预热；不含读图/预处理/GT/指标，含 DIS、原生计算、输出D2H。
一次头调用时 EMA松弛与固定gain是同一代数操作，不算两个独立方法。
额外训练仅RGB0..255，原P继承的既有训练暴露仍保留。旧三视频只作开发，无独立测试或Y11通过。
预测审计数组、GT、缓存、模型和优化器留服务器，review ZIP不包含它们。
''',encoding='utf-8')
    return summary
