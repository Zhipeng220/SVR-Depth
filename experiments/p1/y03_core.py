"""Independent diagnostic suite infrastructure; no training or public-test claims."""
from __future__ import annotations
import argparse, contextlib, csv, fcntl, hashlib, importlib, importlib.util
import json, math, os, statistics, sys, time, traceback, zipfile
from pathlib import Path
from types import SimpleNamespace

SEQS=('crowd2','crowd3','balloon2')
N_RGB={'crowd2':895,'crowd3':854,'balloon2':469}
TAIL={'crowd2':768,'crowd3':256,'balloon2':256}
AGES={16:(4,8,12),32:(8,16,24)}
HORIZONS=(4,8,16)
H=16
TOL=1e-8
HELPER_SHA='6fbe1a2464a7f9756f9ea68cf05f3476a89b005b8e6f889dd1a0093f231c3116'
MODEL_SOURCE={'velodepth/models/velodepth.py':'739c5f27a749174987b9a36edaac32fb98070e87dbef0a0a52f7430d61149479',
 'velodepth/models/encoder_velo.py':'3c5aa2a3a6dce03b68373953037fe8a0313a2cab36d66e5e3b88a56ec4a7ba8a'}

def require(ok,message):
    if not ok:raise RuntimeError(message)
def digest(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):d.update(block)
    return d.hexdigest()
def canonical_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    require(bool(rows),'Refusing an empty result table')
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    tmp.replace(p)
def truth(x):return str(x).strip().lower() in ('true','1','yes')
def quantile(a,q):
    require(bool(a),'Empty quantile');a=sorted(a);x=(len(a)-1)*q;i=int(x)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)

def repeat_features(features,head,q,beta,on_step=None):
    require(isinstance(q,int) and q>=0 and 0.<=beta<=1.,'Invalid repetition/EMA control')
    for j in range(q):
        proposal=head(features)
        if beta==0.:updated=features
        elif beta==1.:updated=proposal
        else:updated=[x+beta*(y-x) for x,y in zip(features,proposal)]
        if on_step is not None:on_step(j+1,updated)
        features=updated
    return features

def sample_cycles(start,end,k):
    candidates=list(range(((start+k-1)//k)*k,end+1,k))
    candidates=[c for c in candidates if c+max(AGES[k])+H<=end]
    require(len(candidates)>=3,f'Insufficient complete cycles: {start},{end},K{k}')
    return [candidates[i] for i in (0,(len(candidates)-1)//2,len(candidates)-1)]

def cases(phase,smoke=False):
    require(phase in ('prefix','tail'),'Unknown split')
    result=[]
    for seq in SEQS:
        for k in (16,32):
            start,end=(0,255) if phase=='prefix' else (TAIL[seq],N_RGB[seq]-1)
            for cycle in sample_cycles(start,end,k):
                for age in AGES[k]:
                    for origin in (('A',) if phase=='prefix' else ('A','P')):
                        anchor=cycle+age
                        result.append(dict(case_id=f'{phase}_{seq}_K{k}_{origin}_t{anchor:06d}',split=phase,
                            sequence=seq,K=k,origin=origin,cycle_start=cycle,age=age,anchor=anchor,
                            snapshot_index=anchor-1,future_end=anchor+H,horizon=H,
                            next_keyframe=(anchor//k+1)*k,numeric_flow_policy='outside_autocast' if origin=='A' else 'inside_autocast'))
    if smoke:
        # Independent prefix-only mechanical smoke. Never call this a tail result.
        result=[]
        for seq in SEQS:
            for origin in ('A','P'):
                result.append(dict(case_id=f'smoke_{seq}_K16_{origin}_t000004',split='smoke',sequence=seq,K=16,
                    origin=origin,cycle_start=0,age=4,anchor=4,snapshot_index=3,future_end=20,horizon=H,
                    next_keyframe=16,numeric_flow_policy='outside_autocast' if origin=='A' else 'inside_autocast'))
        if phase=='prefix':result=[c for c in result if c['origin']=='A']
    return result

def load_existing_unit(folder,signature):
    p=Path(folder)/'complete.json'
    if not p.exists():return None
    obj=read_json(p)
    require(obj['signature']==signature,'Unit identity changed; use original inputs or a new task')
    require(digest(p.with_name('frames.csv'))==obj['rows_sha256'],'Completed unit CSV changed')
    for name,sha in obj.get('audit_files',{}).items():
        require(digest(p.parent/name)==sha,f'Completed audit array missing/changed: {name}')
    return obj

def make_archive(out,prefix,source_paths):
    out=Path(out);archive_path=out/(prefix+'_review.zip');tmp_path=out/(prefix+'_review.zip.tmp')
    with zipfile.ZipFile(tmp_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for item in sorted(out.rglob('*')):
            relative=item.relative_to(out)
            if item.is_file() and item.suffix in ('.json','.jsonl','.csv','.md') and not any(s in relative.parts for s in ('cache','depth','gt')) and item.name!='failure.json':
                z.write(item,str(relative))
        for source_path in source_paths:
            source_path=Path(source_path)
            if source_path.is_file():z.write(source_path,'source/'+source_path.name)
    with zipfile.ZipFile(tmp_path) as z:require(z.testzip() is None,'Review ZIP failed integrity check')
    tmp_path.replace(archive_path)
    require(archive_path.parent==out and archive_path.name==prefix+'_review.zip','Archive destination escaped output')
    return archive_path

def paired_windows(rows,control):
    """Keep every planned frame, including failed/missing-GT rows; never average failures away."""
    by={}
    for r in rows:by.setdefault(r['arm'],{})[int(r['h'])]=r
    require(control in by,'Missing paired control')
    require(all(set(x)==set(range(H+1)) for x in by.values()),'Unpaired or incomplete RGB trajectory')
    result=[]
    for arm,trajectory in by.items():
        for horizon in HORIZONS:
            shared=[h for h in range(1,horizon+1) if truth(by[control][h]['matched'])]
            require(all(truth(trajectory[h]['matched'])==truth(by[control][h]['matched']) for h in range(H+1)),'GT identities diverged')
            # Numerical failure on an RGB frame is fatal even when that frame lacks GT.
            failed=any(not truth(trajectory[h]['valid_prediction']) for h in range(horizon+1))
            ref_failed=any(not truth(by[control][h]['valid_prediction']) for h in range(horizon+1))
            status='NUMERICAL_FAILURE' if failed or ref_failed else ('OK' if shared else 'NO_GT')
            a=[float(trajectory[h]['AbsRel']) for h in shared] if status=='OK' else []
            b=[float(by[control][h]['AbsRel']) for h in shared] if status=='OK' else []
            d=[x-y for x,y in zip(a,b)]
            before=[h for h in shared if not truth(trajectory[h]['after_reset'])]
            result.append(dict(arm=arm,horizon=horizon,status=status,n_future_RGB=horizon,n_GT=len(shared),
                mean_AbsRel=statistics.mean(a) if a else '',sum_AbsRel=sum(a) if a else '',
                mean_RMSE=statistics.mean(float(trajectory[h]['RMSE']) for h in shared) if a else '',
                mean_delta1=statistics.mean(float(trajectory[h]['delta1']) for h in shared) if a else '',
                gain_vs_control=sum(y-x for x,y in zip(a,b)) if a else '',
                harmed_future_frames=sum(x>TOL for x in d) if a else '',
                harmed_fraction=sum(x>TOL for x in d)/len(d) if d else '',
                harm_gt_0p01=sum(x>.01 for x in d) if a else '',
                frame_AbsRel_p95=quantile(a,.95) if a else '',
                n_pre_reset_GT=len(before),
                pre_reset_gain=sum(float(by[control][h]['AbsRel'])-float(trajectory[h]['AbsRel']) for h in before) if a and before else '',
                head_calls=int(trajectory[horizon]['head_calls_cumulative']),
                current_AbsRel=trajectory[0]['AbsRel'],
                current_output_excluded_from_future=True))
    return result

def group_windows(windows,group_keys):
    groups={}
    for row in windows:groups.setdefault(tuple(row[k] for k in group_keys),[]).append(row)
    result=[]
    for key,rr in sorted(groups.items()):
        ok=all(r['status']=='OK' for r in rr)
        out=dict(zip(group_keys,key));out.update(n_cases=len(rr),failed_cases=sum(r['status']!='OK' for r in rr),status='OK' if ok else 'INCOMPLETE_OR_FAILED')
        for m in ('mean_AbsRel','mean_RMSE','mean_delta1','gain_vs_control','harmed_fraction','frame_AbsRel_p95'):
            out[m]=statistics.mean(float(r[m]) for r in rr) if ok else ''
        result.append(out)
    return result

def preflight(args,logic,source_paths):
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root));sys.path.insert(0,str(root/'experiments/p1'))
    hp=root/'experiments/p1/gt4a2_y04a_native_pre_readout_eval.py'
    require(hp.is_file() and digest(hp)==HELPER_SHA,'Reviewed Y04-A helper missing/changed; do not overwrite old sources')
    spec=importlib.util.spec_from_file_location(logic.PREFIX+'_original_helper',hp)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    qa=SimpleNamespace(snapshot=args.snapshot,y04_dir=args.y04_dir,p_checkpoint='',seqs=list(SEQS),smoke_only=False,
        **{s+'_pair_csv':getattr(args,s+'_pair_csv') for s in SEQS})
    base,manifests,_,rt,head_state=helper.preflight(qa)
    require(rt.torch.cuda.is_available(),'CUDA required for real diagnostics')
    for rel,sha in MODEL_SOURCE.items():require(digest(root/rel)==sha,f'Reviewed model source differs: {rel}')
    require(tuple(head_state['blocks.0.0.weight'].shape)==(32,512,1,1),'Expected reviewed 512-channel P head')
    if args.y09_dir=='auto':
        candidates=[]
        for q in (root/'outputs').glob('gt4a2_y08cal_y09pareto_*/validation/summary.json'):
            try:
                data=read_json(q)
                if data.get('status')=='DEVELOPMENT_SUITE_COMPLETE' and data.get('full_three_sequence_evaluation') and not data.get('partial_smoke'):candidates.append(q)
            except (OSError,ValueError):continue
        require(bool(candidates),'Completed Y08-Cal/Y09-Pareto validation required; set Y09_DIR')
        y09=max(candidates,key=lambda q:q.stat().st_mtime_ns).parent.resolve()
    else:y09=Path(args.y09_dir).resolve()
    output=Path(args.out_dir).resolve()
    require(output not in (root,root/'outputs') and y09 not in (output,*output.parents),'Diagnostic output must not overwrite or enter old Y09')
    old=read_json(y09/'protocol.json');old_summary=read_json(y09/'summary.json')
    require(old_summary.get('status')=='DEVELOPMENT_SUITE_COMPLETE' and old_summary.get('quality_RGB_records')==79848,'Y09 reference is incomplete')
    require(digest(y09/'protocol.json')==old_summary['signature'],'Y09 protocol/summary signature differs')
    require(helper.P_SHA in old['input_sha256'].values(),'Y09 did not use reviewed P checkpoint')
    old_env=read_json(y09/'environment.json')
    rt.torch.backends.cudnn.benchmark=old_env['cudnn_benchmark']
    rt.torch.backends.cuda.matmul.allow_tf32=old_env['matmul_allow_tf32']
    rt.torch.backends.cudnn.allow_tf32=old_env['cudnn_allow_tf32']
    refs={}
    for r in read_csv(y09/'y08y09_frame_metrics.csv'):
        if r['variant'] in [f'{f}_K{k}_g{0 if f=="A" else 1}_f1/raw' for f in ('A','P') for k in (16,32)]:
            key=(r['variant'],r['sequence'],int(r['index']));require(key not in refs,'Duplicate Y09 reference key');refs[key]=r
    require(len(refs)==4*sum(N_RGB.values()),'Incomplete shared RGB reference trajectories')
    inputs={str(p.resolve()):digest(p) for p in source_paths}
    inputs[str(hp.resolve())]=HELPER_SHA
    for rel,sha in base['source_sha256'].items():inputs[str((root/rel).resolve())]=sha
    for p in (Path(base['checkpoint']),Path(args.snapshot)/'model.safetensors',Path(args.snapshot)/'config.json',
              Path(args.y04_dir)/'y04_run_identity.json',Path(args.y04_dir)/'y04_summary.json',
              Path(args.y04_dir)/'y04_all_frame_metrics.csv',y09/'protocol.json',y09/'summary.json',y09/'environment.json',y09/'y08y09_frame_metrics.csv'):
        inputs[str(p.resolve())]=digest(p)
    for seq,records in manifests.items():
        require(len(records)==N_RGB[seq],f'RGB manifest length differs: {seq}')
        mp=Path(getattr(args,seq+'_pair_csv')).resolve();inputs[str(mp)]=digest(mp)
        for r in records:
            for name in ('rgb_path','depth_path'):
                if name=='depth_path' and str(r.get('gt_match_status','')).upper()!='MATCHED':continue
                path=Path(r[name]).resolve();require(path.is_file(),f'Data missing: {path}; keep CSV intact, repair compatibility links')
                if str(path) not in inputs:inputs[str(path)]=digest(path)
    props=rt.torch.cuda.get_device_properties(0)
    environment=dict(gpu_name=props.name,uuid=str(getattr(props,'uuid','')),total_memory=props.total_memory,
        torch=str(rt.torch.__version__),cuda=str(rt.torch.version.cuda),python=sys.version.split()[0],
        numpy=rt.np.__version__,opencv=rt.cv2.__version__,threads=4,
        cudnn_benchmark=rt.torch.backends.cudnn.benchmark,matmul_allow_tf32=rt.torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=rt.torch.backends.cudnn.allow_tf32)
    plan=dict(experiment=logic.EXPERIMENT,version=logic.VERSION,partial_smoke=args.smoke_only,
        paths=dict(snapshot=str(Path(args.snapshot).resolve()),y04_dir=str(Path(args.y04_dir).resolve()),y09_dir=str(y09)),
        training_performed=False,checkpoint_sha256=helper.P_SHA,base_weights_sha256=helper.MODEL_SHA,
        K=[16,32],ages={str(k):list(v) for k,v in AGES.items()},horizons=list(HORIZONS),
        tail_cases=cases('tail',args.smoke_only),prefix_cases=cases('prefix',args.smoke_only) if logic.EXPERIMENT=='Y03' else [],
        split='original three development videos only; deterministic first/middle/last complete cycles; no GT-error selection',
        state_contract=logic.state_contract(),arms=logic.plan_description(),
        controls='natural A/P carriers independently replayed; shared RGB/flow-policy and checkpoint within every fork',
        numeric_policy='match each origin: A DIS normalization outside AMP; P inside AMP; all arms in a fork identical',
        save_depth=args.save_depth,depth_audit_h=[0,4,8,16],raw_predictions='float32 NPZ on server; review ZIP omits arrays/cache/GT/weights',
        input_sha256=inputs,environment=environment,source_y09_signature=old_summary['signature'],
        limitations=['mechanism diagnosis, not a deployable new method or Y11 pass','mixed caches in Y10 are explicit off-manifold interventions',
          'selected windows and overlapping futures are correlated; no frame-iid significance or final-test claims',
          'wall-clock includes GT, I/O, audits; not a clean inference latency benchmark'])
    return SimpleNamespace(root=root,rt=rt,helper=helper,base=base,head_state=head_state,manifests=manifests,refs=refs,plan=plan,args=args,logic=logic,y09=y09)

def verify_archive(out,prefix,signature):
    summary=read_json(Path(out)/'summary.json')
    require(summary['signature']==signature,'Result signature differs')
    require(summary['status'] in ('SMOKE_PASS','DIAGNOSIS_COMPLETE_NOT_METHOD_PASS'),'Not a finished result')
    with zipfile.ZipFile(Path(out)/(prefix+'_review.zip')) as z:
        require(z.testzip() is None and json.loads(z.read('summary.json'))==summary,'Review ZIP differs from summary')
    return summary

def main(logic):
    parser=argparse.ArgumentParser(description=logic.DESCRIPTION)
    parser.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT',''))
    parser.add_argument('--y04-dir',default=os.environ.get('Y04_DIR','outputs/gt4a2_y04_validation_fix3_20261025_225902'))
    parser.add_argument('--y09-dir',default=os.environ.get('Y09_DIR','auto'))
    for seq in SEQS:parser.add_argument('--'+seq+'-pair-csv',default=f'outputs/gt4a2_exp1/gt4a2_exp1_{seq}_pairs_used.csv')
    parser.add_argument('--out-dir',default='outputs/gt4a2_'+logic.PREFIX+'_manual/validation')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--save-depth',choices=('audit','all'),default='audit')
    parser.add_argument('--stage',choices=('all','select','diagnose','report') if logic.EXPERIMENT=='Y03' else ('all','diagnose','report'),default='all')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--plan-only',action='store_true');mode.add_argument('--preflight-only',action='store_true');mode.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    if args.plan_only:
        print(json.dumps(dict(experiment=logic.EXPERIMENT,description=logic.DESCRIPTION,state_contract=logic.state_contract(),arms=logic.plan_description(),tail_cases=cases('tail'),prefix_cases=cases('prefix') if logic.EXPERIMENT=='Y03' else []),ensure_ascii=False,indent=2));return 0
    if args.resume and (Path(args.out_dir)/'protocol.json').is_file():
        frozen=read_json(Path(args.out_dir)/'protocol.json')
        require(frozen.get('experiment')==logic.EXPERIMENT,'This checkpoint belongs to a different experiment')
        for field,flag in (('snapshot','--snapshot'),('y04_dir','--y04-dir'),('y09_dir','--y09-dir')):
            if flag in sys.argv:require(str(Path(getattr(args,field)).resolve())==frozen['paths'][field],'Resume source paths must remain frozen')
            setattr(args,field,frozen['paths'][field])
        if '--save-depth' in sys.argv:require(args.save_depth==frozen['save_depth'],'Resume depth storage mode differs')
        args.save_depth=frozen['save_depth']
    require(bool(args.snapshot),'Set VDEEP_SNAPSHOT or pass --snapshot')
    require(not(args.resume and args.smoke_only),'Smoke and formal recovery must use separate output directories')
    out=Path(args.out_dir).resolve()
    require(out.parent.name.startswith('gt4a2_'+logic.PREFIX+'_'),'Use this experiment\'s own outputs/gt4a2_'+logic.PREFIX+'_.../validation namespace')
    out.mkdir(parents=True,exist_ok=True)
    require(Path(args.y04_dir).resolve() not in [out,*out.parents],'New output must be outside old Y04')
    source_dir=Path(__file__).resolve().parent
    source_paths=[source_dir/n for n in (logic.PREFIX+'_core.py',logic.PREFIX+'_runtime.py',logic.PREFIX+'_logic.py',logic.ENTRY,'test_'+logic.PREFIX+'.py')]
    launcher=source_dir.parents[1]/('run_'+logic.PREFIX+'.sh')
    if launcher.exists():source_paths.append(launcher)
    with (out/'.run.lock').open('a+') as tasklock:
        try:fcntl.flock(tasklock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This exact output task is already running')
        had_summary=(out/'summary.json').exists()
        try:
            ctx=preflight(args,logic,source_paths);ctx.out=out
            protocol_path=out/'protocol.json'
            if protocol_path.exists():require(read_json(protocol_path)==ctx.plan,'Input/plan/environment changed; do not edit old protocol.json')
            else:
                require(not args.resume,'Cannot resume without protocol.json');write_json(protocol_path,ctx.plan)
            signature=digest(protocol_path);ctx.signature=signature
            print(f'[PREFLIGHT PASS] {logic.EXPERIMENT}: {len(ctx.plan["tail_cases"])} paired diagnostic cases; frozen P; no training',flush=True)
            if args.preflight_only:return 0
            if (out/'summary.json').exists() and (out/(logic.PREFIX+'_review.zip')).exists():
                verify_archive(out,logic.PREFIX,signature);print('[ALREADY COMPLETE]',out/(logic.PREFIX+'_review.zip'));return 0
            torch=ctx.rt.torch;torch.set_num_threads(4);torch.manual_seed(20261028);torch.cuda.manual_seed_all(20261028)
            uuid=ctx.plan['environment']['uuid'] or ctx.plan['environment']['gpu_name']
            gpu_lock=Path('/tmp')/('svrdepth_diagnostics_'+hashlib.sha256(uuid.encode()).hexdigest()[:16]+'.lock')
            print('[GPU QUEUE] waiting for exclusive diagnostic access',flush=True)
            with gpu_lock.open('a+') as gpulock:
                fcntl.flock(gpulock,fcntl.LOCK_EX)
                runtime=importlib.import_module(logic.PREFIX+'_runtime')
                start=time.perf_counter();result=logic.execute(ctx,runtime)
            if result is None:
                print(f'[STAGE COMPLETE] {args.stage}; continue: bash run_{logic.PREFIX}.sh --resume {out}',flush=True);return 0
            for input_path,expected in ctx.plan['input_sha256'].items():require(digest(input_path)==expected,f'Input changed while running: {input_path}')
            with (out/'invocations.jsonl').open('a') as f:f.write(json.dumps(dict(stage=args.stage,resume=args.resume,diagnostic_wall_sec=time.perf_counter()-start,argv=sys.argv))+'\n')
            summary=dict(experiment=logic.EXPERIMENT,version=logic.VERSION,signature=signature,
                status='SMOKE_PASS' if args.smoke_only else 'DIAGNOSIS_COMPLETE_NOT_METHOD_PASS',partial_smoke=args.smoke_only,
                expected_cases=len(ctx.plan['tail_cases']),completed_cases=len(ctx.plan['tail_cases']),
                training_performed=False,independent_testing_performed=False,Y11_pass=False,
                scientific_conclusion='Read paired contrasts and numerical-failure counts; completion is not mechanism or method PASS',**result)
            write_json(out/'summary.json',summary)
            (out/'RESULT_README.md').write_text(logic.result_note(),encoding='utf-8')
            archive=make_archive(out,logic.PREFIX,source_paths);verify_archive(out,logic.PREFIX,signature)
            (out/'failure.json').unlink(missing_ok=True)
            print(f'[{summary["status"]}] {archive}',flush=True);return 0
        except BaseException as exc:
            if not had_summary:(out/'summary.json').unlink(missing_ok=True)
            write_json(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),
                resume_command=f'bash run_{logic.PREFIX}.sh --resume {out}'))
            raise
