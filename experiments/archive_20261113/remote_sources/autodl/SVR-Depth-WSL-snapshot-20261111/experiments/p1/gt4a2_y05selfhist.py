#!/usr/bin/env python3
"""Frozen-head self-history diagnosis. No optimizer or additional training."""
from y05selfhist_common import *
from types import SimpleNamespace
import argparse,fcntl,os,sys,time,traceback,zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def source_dir(explicit):
    identity=jread(HERE/'y05selfhist_identity.json')
    choices=[Path(explicit).resolve()] if explicit!='auto' else list((ROOT/'outputs').glob('gt4a2_y05_*/validation'))
    found=[]
    for p in choices:
        if (p/'summary.json').is_file() and (p/'protocol.json').is_file():
            s=jread(p/'summary.json')
            if s.get('status')=='Y05_DEVELOPMENT_SUITE_COMPLETE' and not s.get('partial_smoke') and sha(p/'protocol.json')==identity['source_protocol_sha256']:found.append(p)
    require(found,'Reviewed completed Y05 missing. Set Y05_DIR to the ORIGINAL server validation directory with checkpoints.')
    return max(found,key=lambda p:(p/'summary.json').stat().st_mtime_ns).resolve()

def prepare(args):
    identity=jread(HERE/'y05selfhist_identity.json');y05=source_dir(args.y05_dir);oldplan=jread(y05/'protocol.json')
    require(sha(y05/'summary.json')==identity['source_summary_sha256'],'Reviewed Y05 summary differs')
    require(sha(y05/'y05_frame_metrics.csv')==identity['source_frame_metrics_sha256'],'Reviewed Y05 frame table differs')
    for name,digest in identity['y05_source_sha256'].items():
        p=ROOT/name if name.endswith('.sh') else HERE/name
        require(p.is_file() and sha(p)==digest,'Frozen Y05 source missing/changed: '+str(p))
    import y05_common as old
    baseargs=SimpleNamespace(**oldplan['paths'],smoke_only=False)
    ctx=old.preflight(baseargs)
    require(ctx.plan==oldplan,'Y05 source/data/paths/environment changed. Run on original Y05 machine/environment; do not edit protocols.')
    # Original head_states needs BOTH the source training directory and its
    # protocol signature. The new diagnostic signature is assigned by main()
    # only AFTER source checkpoints have passed their original identity checks.
    ctx.out=y05;ctx.signature=sha(y05/'protocol.json')
    from y05_train import head_states
    all_states,selected=head_states(ctx)
    states={h:all_states[h] for h in CARRIERS}
    for h,state in states.items():require(ctx.rt.exp1.state_sha256(state)==identity['head_tensor_sha256'][h],'Trained head identity changed: '+h)
    # Recheck original data/base weights/source at finalization as well as preflight.
    inputs=dict(oldplan['input_sha256'])
    inputs.update({str((HERE/name).resolve()):sha(HERE/name) for name in INSTALLED})
    files=[ROOT/'run_y05selfhist.sh',y05/'summary.json',y05/'protocol.json',y05/'y05_frame_metrics.csv',y05/'training_summary.json',y05/'gradient_audit.json']
    for h in CARRIERS[1:]:files.append(Path(selected[h]['path']))
    for h in CARRIERS[1:]+('P_closed_loop',):files.append(y05/'training'/h/'curve.csv')
    refs={}
    if not args.smoke_only:
        for c in oldplan['probe_plan']:
            if c['origin']!='S8':continue
            folder=y05/'probes'/c['case_id'];meta=jread(folder/'complete.json')
            require(meta['signature']==identity['source_protocol_sha256'] and sha(folder/'frames.csv')==meta['rows_sha256'],'Old P-parent probe changed')
            files += [folder/'complete.json',folder/'frames.csv']
            for r in cread(folder/'frames.csv'):
                action='skip' if r['arm']=='no_pulse' else r['arm']
                if action in ACTIONS and int(r['h'])<=4:refs[c['sequence'],c['anchor'],action,int(r['h'])]=r
    for p in files:inputs[str(p.resolve())]=sha(p)
    cases=anchors(oldplan,args.smoke_only)
    plan=dict(experiment='Y05-SelfHist',version=VERSION,partial_smoke=args.smoke_only,training_performed=False,Y11_pass=False,source_y05_signature=identity['source_protocol_sha256'],paths=dict(y05_dir=str(y05)),environment=oldplan['environment'],carriers=list(CARRIERS),actions=list(ACTIONS),anchors=cases,settings=dict(K=32,period=8,gain=1.,primary_horizon=4,secondary_horizons=[8,16],current_excluded=True,future_policy='after anchor always resume frozen CARRIER head on its normal global S8 schedule; native full keyframe reset',development_only=True,no_tail_selection=True,no_training=True),expected_units=len(cases)*len(CARRIERS)*len(ACTIONS),expected_frame_rows=len(cases)*len(CARRIERS)*len(ACTIONS)*17,budget_audit=budget_audit(y05),input_sha256=inputs)
    ctx.y05=y05;ctx.selfhist_plan=plan;ctx.states=states;ctx.references=refs
    ctx.natural={(r['config_id'],r['sequence'],int(r['index'])):r for r in cread(y05/'y05_frame_metrics.csv')}
    return ctx

def unit_folder(out,carrier,case,action):return out/'units'/carrier/case['case_id']/action

def load_unit(out,signature,carrier,case,action):
    folder=unit_folder(out,carrier,case,action);meta=folder/'complete.json'
    if not meta.exists():return None
    obj=jread(meta);path=folder/'frames.csv'
    require(obj['signature']==signature and obj['rows_sha256']==sha(path),'Completed unit changed: '+str(folder))
    require(obj['carrier']==carrier and obj['action']==action and obj['case']==case and obj['rows']==17,'Completed unit identity changed')
    rows=cread(path);require([int(r['h']) for r in rows]==list(range(17)),'Incomplete branch RGB')
    for row,s in zip(rows,schedule(case['anchor'],carrier,action)):
        require(row['carrier']==carrier and row['action']==action and int(row['anchor'])==case['anchor'] and row['sequence']==case['sequence'],'Unit identity changed')
        require(int(row['index'])==s['index'] and truth(row['repair_applied'])==s['repair_applied'] and row['active_head']==s['active_head'] and truth(row['after_reset'])==s['after_reset'],'Branch continuation policy changed')
    return rows

def compare_metrics(metrics,reference,label):
    require(metrics['valid_pixels']==int(reference['valid_pixels']),'GT mask differs: '+label)
    require(max(abs(float(metrics[m])-float(reference[m])) for m in ('AbsRel','RMSE','delta1'))<=2e-5,'Prediction differs: '+label)

def run_unit(ctx,engine,case,carrier,action,parent,frames,statics):
    from y05_engine import detached_state,numpy_metrics
    torch=ctx.rt.torch
    first=carrier if action=='skip' else action
    engine.head.load_state_dict(ctx.states[first],strict=True)
    state=detached_state(parent,'cuda');rows=[];diagonal=old_parent=0
    for spec in schedule(case['anchor'],carrier,action):
        h=spec['h'];frame=frames[h]
        if h==1:engine.head.load_state_dict(ctx.states[carrier],strict=True)
        calls=engine.gate.calls
        state,pred=engine.eval_step(state,frames[h-1],frame,32,'S8',apply=spec['repair_applied'],static=statics.get(h))
        require(engine.gate.calls-calls==int(spec['repair_applied']),'Actual repair calls disagree with fixed policy')
        metrics=numpy_metrics(ctx.rt.np,pred,frame);pred_sha=hashlib.sha256(pred.tobytes()).hexdigest()
        if carrier==action:
            ref=ctx.natural[carrier+'_K32_S8_g1',case['sequence'],spec['index']]
            require((frame['gt_np'] is not None)==truth(ref['matched']),'Natural diagonal GT status differs')
            if frame['gt_np'] is not None:compare_metrics(metrics,ref,'own-carrier diagonal');diagonal+=1
            if ref['prediction_sha256']:require(pred_sha==ref['prediction_sha256'],'Natural diagonal pixel hash differs')
        key=case['sequence'],case['anchor'],action,h
        if carrier=='P_init' and key in ctx.references:
            ref=ctx.references[key]
            require((frame['gt_np'] is not None)==truth(ref['matched']),'Y05 H4 probe GT differs')
            if frame['gt_np'] is not None:compare_metrics(metrics,ref,'Y05 old P-parent H4');old_parent+=1
        rows.append(dict(case_id=case['case_id'],sequence=case['sequence'],anchor=case['anchor'],age=case['age'],carrier=carrier,action=action,h=h,index=spec['index'],matched=frame['gt_np'] is not None,repair_applied=spec['repair_applied'],active_head=spec['active_head'],after_reset=spec['after_reset'],future_policy='carrier_S8',prediction_sha256=pred_sha,**metrics))
    folder=unit_folder(ctx.out,carrier,case,action);cwrite(folder/'frames.csv',rows)
    jwrite(folder/'complete.json',dict(signature=ctx.signature,carrier=carrier,case=case,action=action,rows=17,rows_sha256=sha(folder/'frames.csv'),diagonal_metric_checks=diagonal,old_P_H4_metric_checks=old_parent))
    return rows

def run_all(ctx):
    from y05_engine import Engine,detached_state,numpy_metrics
    engine=Engine(ctx,SEQS[0]);torch=ctx.rt.torch
    for p in engine.model.parameters():p.requires_grad_(False)
    require(all(not p.requires_grad for p in engine.model.parameters()),'Frozen diagnostic must not enable gradients')
    cases=ctx.selfhist_plan['anchors'];done=0
    for carrier in CARRIERS:
        for case in cases:
            for action in ACTIONS:done+=load_unit(ctx.out,ctx.signature,carrier,case,action) is not None
    try:
        with torch.no_grad():
            for carrier in CARRIERS:
                for seq in SEQS:
                    selected=[c for c in cases if c['sequence']==seq];by_anchor={c['anchor']:c for c in selected}
                    complete=all(load_unit(ctx.out,ctx.signature,carrier,c,a) is not None for c in selected for a in ACTIONS)
                    replay_path=ctx.out/'carrier_replays'/f'{carrier}_{seq}.json'
                    if complete and replay_path.exists():
                        require(jread(replay_path)['signature']==ctx.signature,'Carrier replay signature differs')
                        print(f'[REUSE CARRIER] {carrier}/{seq}',flush=True);continue
                    engine.configure(seq);engine.head.load_state_dict(ctx.states[carrier],strict=True)
                    state=previous=None;checked=0;stop=max(by_anchor)+1
                    for i in range(stop):
                        if i in by_anchor:
                            case=by_anchor[i]
                            if any(load_unit(ctx.out,ctx.signature,carrier,case,a) is None for a in ACTIONS):
                                parent=detached_state(state,'cpu');context={**case,'horizon':16,'origin':'S8'}
                                frames,statics=engine.context(context)
                                for action in ACTIONS:
                                    if load_unit(ctx.out,ctx.signature,carrier,case,action) is None:
                                        run_unit(ctx,engine,case,carrier,action,parent,frames,statics);done+=1
                                        print(f'[SELFHIST] {done}/{ctx.selfhist_plan["expected_units"]} {carrier}/{case["case_id"]}/{action}',flush=True)
                                del parent,frames,statics
                                engine.head.load_state_dict(ctx.states[carrier],strict=True)
                        frame=engine.load(i,i<256 or i>=dict(crowd2=768,crowd3=256,balloon2=256)[seq])
                        state,pred=engine.eval_step(state,previous,frame,32,'S8',apply=pulse_due(i))
                        ref=ctx.natural[carrier+'_K32_S8_g1',seq,i]
                        if frame['gt_np'] is not None:compare_metrics(numpy_metrics(ctx.rt.np,pred,frame),ref,'full carrier replay');checked+=1
                        if ref['prediction_sha256']:require(hashlib.sha256(pred.tobytes()).hexdigest()==ref['prediction_sha256'],'Carrier replay pixel hash differs')
                        previous=frame
                        if i==0 or (i+1)%200==0 or i==stop-1:print(f'[CARRIER] {carrier}/{seq} {i+1}/{stop}',flush=True)
                    jwrite(replay_path,dict(signature=ctx.signature,carrier=carrier,sequence=seq,RGB=stop,natural_metric_checks=checked))
                    del state,previous
    finally:engine.close()

def report(ctx):
    frames=[];all_windows=[];diagonal=old_parent=0
    for carrier in CARRIERS:
        for case in ctx.selfhist_plan['anchors']:
            by={}
            for action in ACTIONS:
                rows=load_unit(ctx.out,ctx.signature,carrier,case,action);require(rows is not None,'Missing planned unit')
                frames+=rows;by[action]={int(r['h']):r for r in rows}
                m=jread(unit_folder(ctx.out,carrier,case,action)/'complete.json')
                diagonal+=m['diagonal_metric_checks'];old_parent+=m['old_P_H4_metric_checks']
            all_windows+=windows(case,carrier,by)
    require(len(frames)==ctx.selfhist_plan['expected_frame_rows'],'Frame record count differs')
    for carrier in CARRIERS:
        for seq in SEQS:
            meta=jread(ctx.out/'carrier_replays'/f'{carrier}_{seq}.json')
            require(meta['signature']==ctx.signature and meta['natural_metric_checks']>0,'Carrier replay canary incomplete')
    require(diagonal>0 and (ctx.selfhist_plan['partial_smoke'] or old_parent>0),'Real diagonal/old-parent canaries missing')
    sequence,macro,comparisons=aggregate(all_windows)
    for name,rows in [('selfhist_frames.csv',frames),('selfhist_future_windows.csv',all_windows),('selfhist_sequence.csv',sequence),('selfhist_macro.csv',macro),('selfhist_vs_C.csv',comparisons),('y05_existing_age_decomposition.csv',age_decomposition(list(ctx.natural.values())))]:cwrite(ctx.out/name,rows)
    jwrite(ctx.out/'source_budget_audit.json',ctx.selfhist_plan['budget_audit'])
    jwrite(ctx.out/'summary.json',dict(experiment='Y05-SelfHist',version=VERSION,signature=ctx.signature,status='SMOKE_PASS' if ctx.selfhist_plan['partial_smoke'] else 'SELFHIST_DIAGNOSIS_COMPLETE',partial_smoke=ctx.selfhist_plan['partial_smoke'],training_performed=False,Y11_pass=False,no_automatic_method_pass=True,carriers=list(CARRIERS),actions=list(ACTIONS),completed_units=ctx.selfhist_plan['expected_units'],frame_records=len(frames),future_windows=len(all_windows),diagonal_metric_checks=diagonal,old_P_H4_metric_checks=old_parent,primary_horizon=4,future_policy='carrier_S8',source_budget_status=ctx.selfhist_plan['budget_audit']['status'],equal_compute_certified=False,numerical_failure_branches=0))
    (ctx.out/'RESULT_README.md').write_text('''# Y05-SelfHist\n\n先读 selfhist_vs_C.csv 的 H4：在每一种相同载体/相同父状态内比较 F/FH 与 C；delta<0 表示比 C 好。再看 selfhist_sequence.csv 的逐视频 skip 相对增益。\nH8/H16 继续执行冻结载体自身的 S8 策略，不能与 Y05 的未来修复关闭窗口混为同一指标。h0 单独保留，始终排除在未来均值之外。\n本实验没有新训练、没有自动选超参数，也不授予 Y11 PASS。不同载体的绝对误差不能直接归因为候选头效果；需要使用共同父状态内的成对差值。重叠窗口不是独立视频样本。\nsource_budget_audit.json 保留旧 P 计时异常，不自动删除或重标等预算检查点。\n''')

def archive(ctx):
    out=ctx.out;archive_path=out/'y05selfhist_review.zip';temp_path=out/'y05selfhist_review.zip.tmp'
    with zipfile.ZipFile(temp_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and (p.suffix in ('.csv','.json','.jsonl','.md')):z.write(p,p.relative_to(out).as_posix())
        for name in INSTALLED:z.write(HERE/name,'source/selfhist/'+name)
        z.write(ROOT/'run_y05selfhist.sh','source/selfhist/run_y05selfhist.sh')
        for name in jread(HERE/'y05selfhist_identity.json')['y05_source_sha256']:
            path=ROOT/name if name.endswith('.sh') else HERE/name
            z.write(path,'source/y05/'+name)
        z.write(ctx.y05/'summary.json','reference/y05_summary.json')
    with zipfile.ZipFile(temp_path) as z:require(z.testzip() is None,'New review ZIP CRC failed')
    temp_path.replace(archive_path)
    return archive_path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--y05-dir',default=os.environ.get('Y05_DIR','auto'))
    p.add_argument('--out-dir',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args();out=Path(args.out_dir).resolve()
    require(out.parent.name.startswith('gt4a2_y05selfhist_'),'Use independent outputs/gt4a2_y05selfhist_... namespace')
    if args.resume:
        frozen=jread(out/'protocol.json');require(frozen['experiment']=='Y05-SelfHist' and not frozen['partial_smoke'],'Only formal SelfHist can resume')
        args.y05_dir=frozen['paths']['y05_dir']
    out.mkdir(parents=True,exist_ok=True);started=time.perf_counter();phase='preflight'
    with (out/'.run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This exact diagnostic is already running')
        try:
            ctx=prepare(args);ctx.out=out;plan=ctx.selfhist_plan
            require(ctx.y05 not in (out,*out.parents),'Output cannot enter old Y05 results')
            path=out/'protocol.json'
            if path.exists():require(args.resume and jread(path)==plan,'Existing/changed plan; use original --resume, never edit protocol')
            else:jwrite(path,plan)
            ctx.signature=sha(path)
            print(f'[PREFLIGHT PASS] {VERSION}: {len(plan["anchors"])} anchors, {plan["expected_units"]} branches; NO TRAINING',flush=True)
            print(f'[SOURCE BUDGET] {plan["budget_audit"]["status"]}; old equal-compute claim is not certified',flush=True)
            if args.preflight_only:return 0
            uuid=plan['environment']['uuid'] or plan['environment']['gpu_name']
            gpu=Path('/tmp')/('svrdepth_diagnostics_'+hashlib.sha256(uuid.encode()).hexdigest()[:16]+'.lock')
            with gpu.open('a+') as gpulock:
                print('[GPU QUEUE] waiting for exclusive access',flush=True);fcntl.flock(gpulock,fcntl.LOCK_EX)
                ctx.rt.torch.set_num_threads(4);phase='carrier-and-counterfactual-replay';run_all(ctx)
                phase='report';report(ctx)
                for input_path,digest in plan['input_sha256'].items():require(sha(input_path)==digest,'Input changed: '+input_path)
                (out/'failure.json').unlink(missing_ok=True)
                with (out/'invocations.jsonl').open('a') as f:f.write(json.dumps(dict(completed=True,resume=args.resume,wall_sec=time.perf_counter()-started,unix_time=time.time(),scope='through table and input verification; archive finalization follows'))+'\n')
                phase='archive';result=archive(ctx);print('[VERIFIED]',result,flush=True)
            return 0
        except BaseException as exc:
            if phase in ('report','archive'):(out/'summary.json').unlink(missing_ok=True)
            jwrite(out/'failure.json',dict(phase=phase,error=str(exc),traceback=traceback.format_exc()))
            with (out/'invocations.jsonl').open('a') as f:f.write(json.dumps(dict(completed=False,resume=args.resume,wall_sec=time.perf_counter()-started,unix_time=time.time()))+'\n')
            raise

if __name__=='__main__':raise SystemExit(main())
