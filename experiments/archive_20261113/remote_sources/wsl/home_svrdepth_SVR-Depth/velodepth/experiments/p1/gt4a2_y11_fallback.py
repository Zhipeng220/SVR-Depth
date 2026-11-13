#!/usr/bin/env python3
"""Y11-A: frozen P_multi K16 confirmation against A/O_fast/O_multi; no training."""
from y11_common import *
from y11_data import source_y09
from types import SimpleNamespace
import argparse,contextlib,fcntl,gc,importlib.util,os,shutil,subprocess,sys,time,traceback,zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def gpu_lock_path():
    """Acquire the shared GPU lock BEFORE CUDA initialization, avoiding queued CUDA clients contaminating current timing."""
    binary=shutil.which('nvidia-smi')
    if not binary and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():binary='/usr/lib/wsl/lib/nvidia-smi'
    require(binary is not None,'nvidia-smi required for verified Y11 timing')
    result=subprocess.run([binary,'--query-gpu=index,uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    inventory=[(a.strip(),b.strip()) for a,b in csv.reader(result.stdout.splitlines())]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0].strip()
    selected=[uuid for index,uuid in inventory if visible in (index,uuid,uuid.removeprefix('GPU-'))]
    require(len(selected)==1,'Cannot map CUDA_VISIBLE_DEVICES to one monitored GPU')
    uuid=selected[0].removeprefix('GPU-')
    return uuid,Path('/tmp')/('svrdepth_diagnostics_'+hashlib.sha256(uuid.encode()).hexdigest()[:16]+'.lock')

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def source_weights(rt,y09,p_state,o_state,identity):
    # Use the OLD source signature here; a new output signature cannot identify old weights.
    path=y09/'training/O_fast/final.pt';obj=rt.torch.load(path,map_location='cpu',weights_only=False)
    source_signature=sha(y09/'protocol.json')
    require(source_signature==identity['y09_protocol_sha256'] and obj['signature']==source_signature,'O_fast source checkpoint signature differs')
    require(rt.exp1.state_sha256(obj['state'])==identity['O_fast_tensor_sha256'],'O_fast tensor identity differs')
    require(all(rt.torch.isfinite(v).all().item() for state in (p_state,o_state,obj['state']) for v in state.values()),'Nonfinite source weights')
    return {'O_fast':obj['state']},path

def prepare(args):
    print('[SOURCE PREFLIGHT] checking original Y04/Y09 sources and checkpoints',flush=True)
    policy=jread(HERE/'y11_policy.json');identity=jread(HERE/'y11_identity.json')
    require(policy['K']==16 and policy['gain']==1 and policy['frequency']==1 and policy['methods']==list(METHODS),'Frozen fallback method changed')
    for rel,digest in identity['source_sha256'].items():require((ROOT/rel).is_file() and sha(ROOT/rel)==digest,'Reviewed runtime source missing/changed: '+rel)
    y09=source_y09(ROOT,args.y09_dir);require(sha(y09/'summary.json')==identity['y09_summary_sha256'],'Reviewed Y09 summary changed')
    oldplan=jread(y09/'protocol.json');old=import_file('y11_original_y09',ROOT/'experiments/p1/gt4a2_y08cal_y09pareto.py')
    # Exact original interface; do not reuse the new output namespace as an old checkpoint namespace.
    oa=SimpleNamespace(snapshot=args.snapshot,y04_dir=args.y04_dir,y04a_dir=args.y04a_dir,smoke_only=False,
        **{s+'_pair_csv':str(ROOT/'outputs/gt4a2_exp1'/f'gt4a2_exp1_{s}_pairs_used.csv') for s in ('crowd2','crowd3','balloon2')})
    source_plan,source_manifests,refs,rt,p_state,o_state,helper=old.preflight(oa)
    oa.protocol=source_plan
    require(source_plan['configuration_plan']==oldplan['configuration_plan'],'Original Y09 configuration identity changed')
    trained,fast_path=source_weights(rt,y09,p_state,o_state,identity)
    oldenv=jread(y09/'environment.json')
    for key in ('cudnn_benchmark','matmul_allow_tf32','cudnn_allow_tf32'):
        target=rt.torch.backends.cuda.matmul if key=='matmul_allow_tf32' else rt.torch.backends.cudnn
        setattr(target,'allow_tf32' if 'allow_tf32' in key else 'benchmark',oldenv[key])
    rt.torch.set_num_threads(4)
    props=rt.torch.cuda.get_device_properties(0)
    environment=dict(gpu_name=props.name,uuid=str(getattr(props,'uuid','')),memory_bytes=props.total_memory,
        torch=str(rt.torch.__version__),cuda=str(rt.torch.version.cuda),numpy=rt.np.__version__,opencv=rt.cv2.__version__,python=sys.version.split()[0],
        threads=4,cudnn_benchmark=rt.torch.backends.cudnn.benchmark,matmul_allow_tf32=rt.torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=rt.torch.backends.cudnn.allow_tf32)
    manifests={};data=None
    if not args.smoke_only:
        data_path=Path(args.dataset_protocol).resolve();data=jread(data_path)
        require(data['policy_sha256']==sha(HERE/'y11_policy.json'),'Dataset was frozen under another policy')
        require(list(data['manifests'])==policy['confirmation_sequences'],'Fixed confirmation recordings differ')
        require(not data['prior_exposure']['conflicts'] and not data['RGB_byte_overlap_with_Y09'],'Known data exposure conflict')
        for p,digest in data['input_sha256'].items():require(Path(p).is_file() and sha(p)==digest,'Frozen confirmation data changed: '+p)
        for path in data.get('expected_missing_depth',[]):require(not Path(path).exists(),'GT availability changed after data freeze: '+path)
        for seq,meta in data['manifests'].items():
            records=rt.exp1.read_manifest(Path(meta['path']),training=False)
            require(len(records)==meta['RGB'],'Manifest count differs');manifests[seq]=records
    inputs=dict(source_plan['input_sha256'])
    inputs.update({str((HERE/n).resolve()):sha(HERE/n) for n in FILES})
    for p in (ROOT/'run_y11.sh',y09/'protocol.json',y09/'summary.json',y09/'environment.json',fast_path):inputs[str(p.resolve())]=sha(p)
    for p in (Path(args.y04_dir)/'y04_run_identity.json',Path(args.y04_dir)/'training/P_multi/gt4a2_c3_C_training_summary.json'):
        inputs[str(p.resolve())]=sha(p)
    for method in METHODS:
        path=y09/'quality'/config(method)['id']/'crowd2.csv';require(path.is_file(),'Original K16 canary table missing '+str(path));inputs[str(path.resolve())]=sha(path)
    if data is not None:inputs.update(data['input_sha256']);inputs[str(Path(args.dataset_protocol).resolve())]=sha(args.dataset_protocol)
    plan=dict(experiment='Y11-Fallback',version=VERSION,stage=policy['stage'],partial_smoke=args.smoke_only,training_performed=False,
        candidate='reviewed P_multi',policy=policy,environment=environment,paths=dict(snapshot=str(Path(args.snapshot).resolve()),y04_dir=str(Path(args.y04_dir).resolve()),y04a_dir=str(args.y04a_dir),y09_dir=str(y09),dataset_protocol=str(Path(args.dataset_protocol).resolve())),
        source_y09_signature=identity['y09_protocol_sha256'],new_training_seeds=0,inherited_training_seed_count=1,
        sequences={s:dict(RGB=len(rows),matched_GT=sum(str(r['gt_match_status']).upper()=='MATCHED' for r in rows)) for s,rows in manifests.items()},
        timing='serial warm-file-cache RGB read/decode/preprocess/flow/state/head/readout/CPU depth/sync; GT/metrics/writes outside timer',
        data_independence=data['independence_scope'] if data else 'smoke uses OLD crowd2 prefix only',
        final_blind_test=False,full_Y11_certification=False,input_sha256=inputs)
    return SimpleNamespace(policy=policy,identity=identity,old=old,oa=oa,rt=rt,p_state=p_state,o_state=o_state,trained=trained,
        y09=y09,manifests=manifests,source_manifests=source_manifests,plan=plan)

def npz(np,path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)

def new_stream(ctx,method,records,debug):
    from y08y09_models import Stream
    cfg=config(method);m,adapter=ctx.old.make_model(ctx.oa,ctx.rt,cfg,ctx.p_state,ctx.o_state,ctx.trained)
    shape=ctx.rt.exp1.shape_info_for(m,Path(records[0]['rgb_path']),ctx.rt.torch.device('cuda'))
    return Stream(ctx.rt,m,cfg,shape,adapter,debug=debug),m,adapter,shape

def cleanup(ctx,*objects):
    gc.collect();ctx.rt.torch.cuda.empty_cache()

def smoke(ctx):
    rt=ctx.rt;records=ctx.source_manifests['crowd2'][:32];reports=[]
    with rt.torch.no_grad():
        for method in METHODS:
            stream,m,adapter,shape=new_stream(ctx,method,records,True);checks=0
            refs={int(r['index']):r for r in cread(ctx.y09/'quality'/config(method)['id']/'crowd2.csv')}
            for i,rec in enumerate(records):
                pred,meta=stream.step(i,rec);require(meta['keyframe']==(i%16==0) and meta['age']==i%16,'Smoke fixed schedule differs')
                if str(rec['gt_match_status']).upper()=='MATCHED':
                    gt=rt.exp1.load_frame(rec,shape,rt.torch.device('cuda'),need_gt=True)['gt_np'];mm=metric(rt.np,pred,gt);ref=refs[i]
                    require(mm['valid_pixels']==int(ref['valid_pixels']),'Old smoke GT mask differs')
                    require(max(abs(mm[k]-float(ref[k])) for k in ('AbsRel','RMSE','delta1'))<=2e-5,'Original K16 prediction changed: '+method);checks+=1
            require(stream.counts==expected_counts(method,len(records)),'Smoke actual model call count differs')
            reports.append(dict(method=method,RGB=len(records),original_metric_checks=checks,counts=stream.counts))
            del stream,m,adapter;cleanup(ctx)
    require(all(r['original_metric_checks']>0 for r in reports),'Smoke lacked real GT checks')
    jwrite(ctx.out/'smoke_checks.json',dict(signature=ctx.signature,passed=True,reports=reports,no_new_recording_predictions=True))
    print('[SMOKE PASS] four exact K16 source trajectories',flush=True)

def quality_path(out,method,seq):return out/'quality'/method/(seq+'.csv')
def load_quality(ctx,method,seq):
    path=quality_path(ctx.out,method,seq);meta_path=path.with_suffix('.json')
    if not meta_path.exists():return None
    meta=jread(meta_path);require(meta['signature']==ctx.signature and meta['rows_sha256']==sha(path),'Quality unit changed')
    for p,digest in meta['binary_sha256'].items():require(Path(p).is_file() and sha(p)==digest,'Saved prediction/GT changed: '+p)
    rr=cread(path);require(len(rr)==len(ctx.manifests[seq]) and [int(r['index']) for r in rr]==list(range(len(rr))),'Quality trajectory incomplete')
    for i,(r,record) in enumerate(zip(rr,ctx.manifests[seq])):
        require(r['method']==method and r['sequence']==seq and truth(r['matched'])==(str(record['gt_match_status']).upper()=='MATCHED'),'Quality identity/GT selection differs')
        require(truth(r['keyframe'])==(i%16==0) and int(r['age'])==i%16,'Saved fixed schedule differs')
    require(meta['counts']==expected_counts(method,len(rr)),'Saved actual counts differ')
    return rr

def gt_for(ctx,seq,i,rec,shape):
    rt=ctx.rt;path=ctx.out/'gt'/seq/f'{i:06d}.npz'
    if path.exists():
        with rt.np.load(path) as z:return z['depth'].copy(),path
    raw=rt.cv2.imread(str(rec['depth_path']),rt.cv2.IMREAD_UNCHANGED)
    require(raw is not None and raw.dtype==rt.np.uint16 and raw.ndim==2 and tuple(raw.shape)==tuple(shape['original_shape']),'Registered uint16 GT geometry differs')
    gt=raw.astype(rt.np.float32)/ctx.policy['depth_scale'];mask=(gt>0)&rt.np.isfinite(gt)
    require(bool(mask.any()),'Empty GT mask');npz(rt.np,path,depth=gt,valid_mask=mask);return gt,path

def quality_unit(ctx,method,seq):
    previous=load_quality(ctx,method,seq)
    if previous is not None:print('[REUSE QUALITY]',method,seq,flush=True);return previous
    rt=ctx.rt;records=ctx.manifests[seq];stream,m,adapter,shape=new_stream(ctx,method,records,True)
    rr=[];binaries={};base={}
    if method!='A':base={int(r['index']):r for r in load_quality(ctx,'A',seq)}
    with rt.torch.no_grad():
        for i,rec in enumerate(records):
            pred,meta=stream.step(i,rec)
            require(tuple(pred.shape)==tuple(shape['original_shape']),'Prediction geometry differs')
            require(meta['keyframe']==(i%16==0) and meta['age']==i%16,'Fixed K16 changed')
            matched=str(rec['gt_match_status']).upper()=='MATCHED'
            row=dict(method=method,sequence=seq,index=i,rgb_timestamp=rec['rgb_timestamp'],rgb_file=Path(rec['rgb_path']).name,
                gt_file=Path(rec['depth_path']).name if matched else '',matched=matched,gt_match_status=rec['gt_match_status'],keyframe=meta['keyframe'],age=meta['age'],
                repair_requested=meta['repair_requested'],output_adapter_applied=meta['output_adapter_applied'],
                prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),AbsRel='',RMSE='',delta1='',valid_pixels=0,mask_sha256='')
            require(bool((rt.np.isfinite(pred)&(pred>0)).all()),'Invalid prediction on RGB; no silent frame drops')
            pred_path=ctx.out/'predictions'/method/seq/f'{i:06d}.npz';npz(rt.np,pred_path,depth=pred,index=rt.np.array(i),rgb_timestamp=rt.np.array(float(rec['rgb_timestamp'])));binaries[str(pred_path)]=sha(pred_path)
            if matched:
                gt,gpath=gt_for(ctx,seq,i,rec,shape);binaries[str(gpath)]=sha(gpath);row.update(metric(rt.np,pred,gt))
                if base:
                    require(truth(base[i]['matched']) and row['mask_sha256']==base[i]['mask_sha256'],'Common mask differs')
                if method.startswith('O_'):
                    mm=metric(rt.np,stream.native_prediction.detach().float().cpu().numpy(),gt)
                    require(max(abs(mm[k]-float(base[i][k])) for k in ('AbsRel','RMSE','delta1'))<=2e-5,'Output-only adapter changed native A trajectory')
            rr.append(row)
            if i==0 or (i+1)%200==0 or i+1==len(records):print('[QUALITY]',method,seq,f'{i+1}/{len(records)}',flush=True)
    require(stream.counts==expected_counts(method,len(records)),'Actual quality call count differs')
    path=quality_path(ctx.out,method,seq);cwrite(path,rr)
    jwrite(path.with_suffix('.json'),dict(signature=ctx.signature,rows_sha256=sha(path),counts=stream.counts,binary_sha256=binaries,original_shape=list(shape['original_shape'])))
    del stream,m,adapter;cleanup(ctx);return rr

def timing_path(out,method,seq,repeat):return out/'timing'/method/f'{seq}_r{repeat}.json'
def load_timing(ctx,method,seq,repeat):
    path=timing_path(ctx.out,method,seq,repeat)
    if not path.exists():return None
    meta=jread(path);require(meta['signature']==ctx.signature and sha(path.with_suffix('.csv'))==meta['rows_sha256'],'Timing unit changed')
    require(meta['endpoint_clean'] and meta['quality_checked'],'Unverified timing cannot be reused');return meta

def timing_unit(ctx,method,seq,repeat,position):
    previous=load_timing(ctx,method,seq,repeat)
    if previous is not None:print('[REUSE TIMING]',method,seq,repeat,flush=True);return previous
    rt=ctx.rt;records=ctx.manifests[seq];qrows={int(r['index']):r for r in load_quality(ctx,method,seq)}
    stream,m,adapter,shape=new_stream(ctx,method,records,False)
    rr=[];checks=0
    with rt.torch.no_grad():
        for rec in records:Path(rec['rgb_path']).read_bytes()
        for i,rec in enumerate(records[:min(ctx.policy['timing_warmup_RGB'],len(records))]):stream.step(i,rec)
        rt.torch.cuda.synchronize();stream.reset();before=ctx.old.gpu_endpoint(rt)
        require(before['clean'],'Timing GPU endpoint not clean; no timing accepted: '+str(before))
        rt.torch.cuda.reset_peak_memory_stats()
        for i,rec in enumerate(records):
            rt.torch.cuda.synchronize();start=time.perf_counter();pred,meta=stream.step(i,rec);rt.torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            rr.append(dict(method=method,sequence=seq,index=i,repeat=repeat,position=position,latency_ms=elapsed,keyframe=meta['keyframe'],age=meta['age']))
            require(meta['keyframe']==truth(qrows[i]['keyframe']) and meta['age']==int(qrows[i]['age']),'Timing schedule differs from quality')
            if i in (0,len(records)-1) or i%16 in (1,15):
                with rt.np.load(ctx.out/'predictions'/method/seq/f'{i:06d}.npz') as z:expected=z['depth']
                require(bool(rt.np.allclose(pred,expected,rtol=1e-5,atol=1e-4)),'Timing prediction differs from quality');checks+=1
            if i==0 or (i+1)%200==0 or i+1==len(records):print('[TIMING]',f'r{repeat}',method,seq,f'{i+1}/{len(records)}',flush=True)
        after=ctx.old.gpu_endpoint(rt)
    require(stream.counts==expected_counts(method,len(records)),'Timing call counts differ')
    path=timing_path(ctx.out,method,seq,repeat);cwrite(path.with_suffix('.csv'),rr)
    valid=bool(before['clean'] and after['clean'])
    meta=dict(signature=ctx.signature,method=method,sequence=seq,repeat=repeat,position=position,RGB=len(records),
        whole_mean_ms=statistics.mean(r['latency_ms'] for r in rr),whole_p50_ms=quantile([r['latency_ms'] for r in rr],.5),whole_p95_ms=quantile([r['latency_ms'] for r in rr],.95),
        peak_allocated_gib=rt.torch.cuda.max_memory_allocated()/1024**3,endpoint_clean=valid,endpoints=[before,after],quality_checked=checks>0,depth_checks=checks,
        counts=stream.counts,rows_sha256=sha(path.with_suffix('.csv')),cache='warm file bytes only; decode/preprocess/flow/state/readout/CPU depth timed')
    if not valid:
        jwrite(path.with_name(path.stem+f'_rejected_{time.time_ns()}.json'),meta)
        raise RuntimeError('GPU contention at timing endpoint; rejected evidence retained, rerun this unit with --resume')
    jwrite(path,meta);del stream,m,adapter;cleanup(ctx);return meta

def report(ctx):
    frames=[];timing=[]
    for method in METHODS:
        for seq in ctx.policy['confirmation_sequences']:
            rows=load_quality(ctx,method,seq);require(rows is not None,'Missing planned quality unit');frames+=rows
            for repeat in range(ctx.policy['timing_repeats']):
                obj=load_timing(ctx,method,seq,repeat);require(obj is not None,'Missing planned timing unit');timing.append(obj)
    sequence,macro,dominance,intervals,decision=aggregate(frames,timing,ctx.policy)
    for name,rows in [('y11_frame_metrics.csv',frames),('y11_sequence_quality.csv',sequence),('y11_macro_quality.csv',macro),('y11_observed_dominance.csv',dominance),('y11_recording_bootstrap.csv',intervals)]:cwrite(ctx.out/name,rows)
    jwrite(ctx.out/'y11_candidate_decision.json',decision)
    jwrite(ctx.out/'summary.json',dict(experiment='Y11-Fallback',version=VERSION,status='Y11_A_FIXED_CANDIDATE_CONFIRMATION_COMPLETE',signature=ctx.signature,partial_smoke=False,
        training_performed=False,candidate='reviewed P_multi',candidate_changed=False,K=16,gain=1.,frequency=1,
        quality_units=len(METHODS)*len(ctx.manifests),quality_RGB_records=len(frames),timing_trials=len(timing),all_timing_verified=True,
        all_GT_masks_checked=True,numerical_failure_units=0,inherited_training_seeds=1,new_training_seeds=0,
        working_point_supported_on_this_recording_set=decision['working_point_supported_on_this_recording_set'],Y11_pass=False,
        final_blind_test=False,independence_scope=ctx.policy['independence_scope'],M01_M06_started=False,
        interpretation='Y11-A one fixed checkpoint on new Bonn recordings; completion is not full Y11/physical-scene/multi-seed/final-test PASS'))
    (ctx.out/'RESULT_README.md').write_text('''# Y11-A 备用路线独立录制候选确认\n\n先读 y11_candidate_decision.json 的每个预先冻结判据，再看逐视频质量、严重伤害、实测端到端耗时和八轴 observed dominance。所有方法均K16、raw、gain1、逐传播帧，无新训练、无校准、无新候选选择。\n图表使用新轮次统一实测成本，不拼接旧Y09或Y05绝对时延。三个计时重复不是三个训练种子。场景宏为录制序列等权；bootstrap只描述本集合录制间差异，Bonn新录制不等同不同物理场景。\n本阶段成功完成不自动授予完整Y11 PASS，也没有开始M01–M06。原服务器保留逐帧float32预测、GT和固定mask；review包只含表、哈希和协议，避免上传大型张量。\n''',encoding='utf-8')

def archive(ctx):
    target=ctx.out/'y11_review.zip';temp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ctx.out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.csv','.md','.jsonl') and p.name!='failure.json':z.write(p,p.relative_to(ctx.out).as_posix())
        for name in FILES:z.write(HERE/name,'source/y11/'+name)
        z.write(ROOT/'run_y11.sh','source/y11/run_y11.sh')
        for rel in ctx.identity['source_sha256']:z.write(ROOT/rel,'source/repository/'+rel)
        if hasattr(ctx,'y09'):
            for name in ('protocol.json','summary.json','environment.json'):z.write(ctx.y09/name,'reference/y09_'+name)
        if 'dataset_protocol' in ctx.plan.get('paths',{}):
            dataset=Path(ctx.plan['paths']['dataset_protocol']);z.write(dataset,'data/dataset_protocol.json')
            for seq,meta in jread(dataset)['manifests'].items():
                z.write(meta['path'],'data/manifests/'+seq+'.csv')
                if meta.get('data_integrity_audit'):z.write(meta['data_integrity_audit'],'data/data_integrity/'+seq+'.json')
    with zipfile.ZipFile(temp) as z:require(z.testzip() is None,'Review ZIP CRC failed')
    temp.replace(target);return target

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT',''))
    p.add_argument('--y04-dir',default=os.environ.get('Y04_DIR','outputs/gt4a2_y04_validation_fix3_20261025_225902'))
    p.add_argument('--y04a-dir',default=os.environ.get('Y04A_DIR','auto'));p.add_argument('--y09-dir',default=os.environ.get('Y09_DIR','auto'))
    p.add_argument('--dataset-protocol',default='outputs/y11_confirmation_data/dataset_protocol.json');p.add_argument('--out-dir',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args(argv);out=Path(args.out_dir).resolve();require(out.parent.name.startswith('gt4a2_y11_fallback_'),'Use independent Y11 output namespace')
    if args.resume:
        frozen=jread(out/'protocol.json');require(frozen['experiment']=='Y11-Fallback' and not frozen['partial_smoke'],'Only formal Y11-A can resume')
        for k,v in frozen['paths'].items():setattr(args,k,v)
    out.mkdir(parents=True,exist_ok=True);phase='preflight';started=time.perf_counter();gpu_handle=None
    with (out/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            monitored_uuid,gpu=gpu_lock_path();gpu_handle=gpu.open('a+')
            print('[GPU QUEUE] waiting before CUDA initialization',flush=True);fcntl.flock(gpu_handle,fcntl.LOCK_EX)
            require(bool(args.snapshot),'Set VDEEP_SNAPSHOT');ctx=prepare(args);ctx.out=out;path=out/'protocol.json'
            if path.exists():require(args.resume and jread(path)==ctx.plan,'Output protocol differs; never edit it after results')
            else:jwrite(path,ctx.plan)
            ctx.signature=sha(path);print('[PREFLIGHT PASS]',VERSION,'four frozen methods; NO TRAINING',flush=True)
            if args.preflight_only:return 0
            require(ctx.plan['environment']['uuid'].removeprefix('GPU-')==monitored_uuid,'CUDA GPU differs from locked GPU')
            with contextlib.nullcontext(gpu_handle):
                if args.smoke_only:
                    phase='CUDA-smoke';smoke(ctx);jwrite(out/'summary.json',dict(experiment='Y11-Fallback',signature=ctx.signature,status='SMOKE_PASS',partial_smoke=True,Y11_pass=False));return 0
                # Conservative uncompressed storage estimate; all prediction arrays stay on server.
                first=next(iter(ctx.manifests.values()))[0];image=ctx.rt.cv2.imread(first['rgb_path']);require(image is not None,'Cannot inspect RGB dimensions')
                missing_predictions=sum(not (out/'predictions'/method/seq/f'{i:06d}.npz').exists() for method in METHODS for seq,records in ctx.manifests.items() for i in range(len(records)))
                missing_gt=sum(str(r['gt_match_status']).upper()=='MATCHED' and not (out/'gt'/seq/f'{i:06d}.npz').exists() for seq,records in ctx.manifests.items() for i,r in enumerate(records))
                estimate=(4*missing_predictions+5*missing_gt)*image.shape[0]*image.shape[1]
                require(shutil.disk_usage(out).free>estimate+(2<<30),'Insufficient output disk for full predictions/GT; use a data-disk repository')
                phase='quality'
                for method in METHODS:
                    for seq in ctx.policy['confirmation_sequences']:quality_unit(ctx,method,seq)
                phase='timing'
                for repeat in range(ctx.policy['timing_repeats']):
                    for si,seq in enumerate(ctx.policy['confirmation_sequences']):
                        shift=(repeat+si)%len(METHODS);order=METHODS[shift:]+METHODS[:shift]
                        for position,method in enumerate(order):timing_unit(ctx,method,seq,repeat,position)
                phase='report';report(ctx)
                for input_path,digest in ctx.plan['input_sha256'].items():require(sha(input_path)==digest,'Input changed during run: '+input_path)
                (out/'failure.json').unlink(missing_ok=True)
                with (out/'invocations.jsonl').open('a') as f:f.write(json.dumps(dict(completed=True,resume=args.resume,wall_sec=time.perf_counter()-started,scope='through table/input verification; archive follows'))+'\n')
                phase='archive';result=archive(ctx);print('[VERIFIED]',result,flush=True)
            return 0
        except BaseException as exc:
            if phase in ('report','archive'):(out/'summary.json').unlink(missing_ok=True)
            jwrite(out/'failure.json',dict(phase=phase,error=str(exc),traceback=traceback.format_exc()));raise
        finally:
            if gpu_handle is not None:gpu_handle.close()

if __name__=='__main__':raise SystemExit(main())
