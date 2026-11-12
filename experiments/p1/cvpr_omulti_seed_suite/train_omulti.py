#!/usr/bin/env python3
"""Train one Y04 O_multi seed matched to an already completed P_multi seed."""
import argparse,csv,hashlib,importlib,json,os,random,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from seed_contract import SEEDS,ORDER_SHA,BASE_SHA,HISTORICAL_P_SHA,SLOTS,require,sha,order,order_sha,check_p_metadata,supervision_manifest,training_inputs,incomplete_pair

PINS={
 'experiments/p1/gt4a2_y04_fair_multiscene.py':'2a24e5a67487d55177e3374f9c6be5e92795b508dfcdb299cefe46e37cc31d4f',
 'experiments/p1/gt4a2_c3_closed_loop.py':'76327208acac4543b2a87ccba8b7af435e4c5ce65267b42052e51b435cbad7f5',
 'experiments/p1/gt4a2_q0_3_necessity_controls.py':'5ac9e648dbae36e17ebf64cd8891c226052f1a71171acf326f7dc6b591ff2390',
 'experiments/p1/gt4a2_exp1_train_eval.py':'c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f',
 'experiments/p1/gt2_fixed_clip_repair_train.py':'bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192',
 'velodepth/models/decoder_velo.py':'35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08',
 'velodepth/models/persistent_feature_repair.py':'6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c',
 'experiments/p1/gt4a2_y11_fallback.py':'e4c5d46cbe48b56c1631090959803f9d8db4690e6426aa60f8ff68318e0503ce',
 'experiments/p1/y11_common.py':'426e461dc296ecae05cd1783dc5ae42689eaedd1cfdc81906eee75fd50e0222b',
 'experiments/p1/y11_data.py':'4afea4f740d880aed26b055734ffeed1d7efea409f1591f99cdedb2e4a1487bc',
 'experiments/p1/gt4a2_y08cal_y09pareto.py':'ab680c054b3a9c09f6c935c2e4fb12c9995880207145e5adc8ba865ee0cf765e',
 'experiments/p1/y08y09_common.py':'17756ccfee812a2c973be6d7c4905a1d71e42a434f39c25632c0dad76f91ccf3',
}

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def self_test():
    for s in SEEDS:require(order_sha(s)==ORDER_SHA[s],'Sample order hash mismatch')
    print('[SELFTEST PASS] balanced Y04 order, no Torch/CUDA/training',flush=True)

def parser():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--self-test',action='store_true');ap.add_argument('--seed',type=int,choices=SEEDS)
    ap.add_argument('--p-checkpoint');ap.add_argument('--p-sha256');ap.add_argument('--p-summary');ap.add_argument('--p-budget')
    ap.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT',''));ap.add_argument('--out-dir')
    ap.add_argument('--crowd2-pair-csv',default='outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv')
    ap.add_argument('--crowd3-pair-csv',default='outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv')
    ap.add_argument('--balloon2-pair-csv',default='outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv')
    ap.add_argument('--previous-config',default='outputs/gt4a2_exp1/gt4a2_exp1_config.json')
    ap.add_argument('--c3-frame-csv',default='outputs/gt4a2_c3/gt4a2_c3_full_sequence_frame_metrics.csv')
    ap.add_argument('--c3-output-dir',default='outputs/gt4a2_c3')
    ap.add_argument('--c-checkpoint',default='outputs/gt4a2_c3/gt4a2_c3_C_final_step1000.pt')
    ap.add_argument('--d-checkpoint',default='outputs/gt4a2_c3/gt4a2_c3_D_final_step1000.pt')
    modes=ap.add_mutually_exclusive_group();modes.add_argument('--preflight-only',action='store_true');modes.add_argument('--smoke-only',action='store_true')
    ap.add_argument('--resume',action='store_true');return ap

def run(args):
    root=Path.cwd().resolve();require((root/'velodepth/models').is_dir(),'Run from VeloDepth repository root')
    require(args.seed in SEEDS and args.p_checkpoint and args.p_sha256 and args.p_summary and args.snapshot,'seed, P checkpoint/SHA/summary and snapshot are required')
    require(not(args.resume and (args.smoke_only or args.preflight_only)),'Resume is for formal training only')
    require(args.seed!=20261025 or args.smoke_only or args.preflight_only,'Historical seed already exists; use it only for smoke/preflight')
    if not args.smoke_only and not args.preflight_only:require(bool(args.p_budget),'Formal run requires completed P budget report')
    snapshot=Path(args.snapshot).resolve();p_cp=Path(args.p_checkpoint).resolve();p_summary=Path(args.p_summary).resolve()
    require((snapshot/'config.json').is_file() and (snapshot/'model.safetensors').is_file() and sha(snapshot/'model.safetensors')==BASE_SHA,'Snapshot is missing or mismatched')
    require(p_cp.is_file() and sha(p_cp)==args.p_sha256.lower(),'P checkpoint SHA mismatch')
    require(p_summary.is_file(),'P training summary missing');ps=json.loads(p_summary.read_text(encoding='utf-8-sig'))
    require(ps.get('checkpoint_sha256')==sha(p_cp),'P summary/checkpoint SHA mismatch')
    for rel,expected in PINS.items():
        p=root/rel;require(p.is_file() and sha(p)==expected,'Reviewed source missing/changed: '+rel)
    require(order_sha(args.seed)==ORDER_SHA[args.seed],'Seed order invariant failed')
    path=Path(args.out_dir or f'outputs/cvpr_omulti_seed_{args.seed}'+('_smoke' if args.smoke_only else '')).resolve()
    require(path.is_relative_to((root/'outputs').resolve()) and path!=(root/'outputs').resolve(),'Output must be a new dedicated directory under outputs')
    if path.exists() and not (path/'y04_run_identity.json').is_file():
        require(path.is_dir() and not any(path.iterdir()),'Existing output directory has no matching run identity; refuse to write')
    marker=path/'y04_summary.json';require(not marker.exists(),'Output is already completed; do not overwrite')
    sys.path[:0]=[str(root),str(root/'experiments/p1')]
    y04=importlib.import_module('gt4a2_y04_fair_multiscene');y04.load_runtime();y04.SEED=args.seed
    a=SimpleNamespace(snapshot=str(snapshot),crowd2_pair_csv=args.crowd2_pair_csv,crowd3_pair_csv=args.crowd3_pair_csv,balloon2_pair_csv=args.balloon2_pair_csv,
        previous_config=args.previous_config,c3_frame_csv=args.c3_frame_csv,c3_output_dir=args.c3_output_dir,c_checkpoint=args.c_checkpoint,d_checkpoint=args.d_checkpoint,
        out_dir=str(path),resume=bool(args.resume),smoke_only=bool(args.smoke_only),steps=1000)
    qa=y04.args_for_q03(a);pre,manifests,_,_,_,context=y04.q03.build_preflight(qa,list(y04.SEQUENCES))
    pair_paths={s:Path(getattr(a,s+'_pair_csv')).resolve() for s in y04.SEQUENCES};pair_shas={s:sha(p) for s,p in pair_paths.items()}
    obj=y04.torch.load(p_cp,map_location='cpu',weights_only=False)
    require(sha(p_cp)==args.p_sha256.lower(),'P checkpoint changed before or during loading')
    check_p_metadata(obj,args.seed,args.p_sha256.lower(),pair_shas,context['initial_head_hash'])
    if 'y04_outer_seed' in ps:require(int(ps['y04_outer_seed'])==args.seed,'P summary outer seed differs')
    if 'y04_sample_order_sha256' in ps:require(ps['y04_sample_order_sha256']==order_sha(args.seed),'P summary order differs')
    require(y04.balanced_order(24,1000,args.seed)==order(args.seed),'Y04 order differs')
    p_budget=None
    if args.p_budget:
        p_budget=json.loads(Path(args.p_budget).read_text(encoding='utf-8-sig'))
        require(int(p_budget.get('y04_outer_seed',-1))==args.seed and p_budget.get('sample_order_sha256')==order_sha(args.seed),'P budget seed/order mismatch')
        require(int(p_budget.get('multi_scene_optimizer_steps',-1))==1000,'P budget is not 1000 steps')
    rows,supervision_sha=supervision_manifest(manifests)
    fingerprints=training_inputs(manifests)
    require(all(fingerprints.get(r['depth_path'])==r['depth_sha256'] for r in rows),'Training GT changed while building supervision provenance')
    input_signature=hashlib.sha256(json.dumps(fingerprints,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    full_counts=[sum(r['sequence']==s and c*32+1<=r['index']<=c*32+31 for r in rows) for s,c in SLOTS]
    exposure=sum(full_counts[i] for i in order(args.seed))
    if p_budget is not None:require(int(p_budget['sampled_gt_frame_exposures'])==exposure,'P budget GT exposures differ from exact source manifests/order')
    identity=dict(version='CVPR-OMULTI-SEED-1.0',seed=args.seed,smoke=bool(args.smoke_only),source_sha256={x:sha(root/x) for x in PINS},
        runner_sha256=sha(__file__),contract_sha256=sha(Path(__file__).with_name('seed_contract.py')),snapshot_sha256=BASE_SHA,snapshot_config_sha256=sha(snapshot/'config.json'),
        P_checkpoint=str(p_cp),P_checkpoint_sha256=sha(p_cp),P_summary_sha256=sha(p_summary),P_budget_sha256=sha(Path(args.p_budget)) if args.p_budget else None,
        initial_head_sha256=context['initial_head_hash'],pair_sha256=pair_shas,order_sha256=order_sha(args.seed),supervision_sha256=supervision_sha,training_input_sha256=input_signature,
        selection_rule='fixed_final_step',train_scope='same 8 cycles each from crowd2/crowd3/balloon2')
    path.mkdir(parents=True,exist_ok=True);y04.ensure_run_identity(path/'y04_run_identity.json',identity,resume=args.resume)
    write(path/'training_input_sha256.json',fingerprints)
    write(path/'preflight.json',dict(status='PREFLIGHT_PASS',identity=identity,P_budget_matched=bool(p_budget),matched_unique_GT=len(rows),expected_GT_exposures=exposure,
        no_evaluation=True,training_performed=False,supervision_hash_definition='canonical JSON of sequence,index,absolute GT path,GT bytes SHA; derived from identical pinned pairing manifests'))
    with (path/'supervision_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sequence','index','depth_path','depth_sha256']);w.writeheader();w.writerows(rows)
    print('[PREFLIGHT PASS] matched P seed',args.seed,'order',order_sha(args.seed),'GT exposures',exposure,flush=True)
    if args.preflight_only:return 0
    from gt4a2_y11_fallback import gpu_lock_path
    import fcntl
    uuid,lockpath=gpu_lock_path();stage='waiting_for_gpu';attempt=dict(seed=args.seed,smoke=bool(args.smoke_only),started_at_utc=datetime.now(timezone.utc).isoformat(),status='STARTED')
    attempts=path/'attempts.jsonl'
    with attempts.open('a',encoding='utf-8') as f:f.write(json.dumps(attempt)+'\n');f.flush();os.fsync(f.fileno())
    phase_started=time.perf_counter()
    try:
        with lockpath.open('a+') as lock:
            print('[GPU QUEUE] waiting for shared GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            t=y04.torch;np=y04.np
            require(t.cuda.is_available() and str(getattr(t.cuda.get_device_properties(0),'uuid','')).removeprefix('GPU-')==uuid,'CUDA device differs from shared lock')
            t.set_num_threads(4);t.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed);t.cuda.manual_seed_all(args.seed)
            device=t.device('cuda');stage='training_cache';started=time.perf_counter();phase_started=started
            # Restrict this arm's cache construction to the 8 matched cycles per sequence.
            original=dict(y04.TRAIN_CYCLES)
            try:
                y04.TRAIN_CYCLES=dict.fromkeys(y04.SEQUENCES,8)
                readers,cache_meta=y04.build_caches(a,context,manifests,path,device,args.smoke_only)
            finally:y04.TRAIN_CYCLES=original
            cache_sec=time.perf_counter()-started;attempt['cache_seconds']=cache_sec;write(path/'cache_manifest.json',cache_meta)
            slots=[('crowd2',0),('crowd3',0),('balloon2',0)] if args.smoke_only else SLOTS
            schedule=[0,1,2] if args.smoke_only else order(args.seed);steps=len(schedule)
            reader=y04.SceneCycleReader(readers,slots);per_slot=[int(readers[s].load(c)['n_supervised']) for s,c in slots]
            expected=[sum(r['sequence']==s and c*32+1<=r['index']<=c*32+31 for r in rows) for s,c in slots]
            require(per_slot==expected,'Cached valid supervision differs from fixed GT frame list')
            actual_exposure=sum(per_slot[i] for i in schedule);attempt['scheduled_GT_exposures']=actual_exposure
            if not args.smoke_only:require(actual_exposure==exposure,'O_multi GT exposure differs from matched P budget')
            stage='train_O_multi';t.cuda.synchronize();t.cuda.reset_peak_memory_stats();started=time.perf_counter();phase_started=started
            from gt4a2_y08cal_y09pareto import gpu_endpoint
            rt=SimpleNamespace(torch=t);before=gpu_endpoint(rt);attempt['gpu_endpoint_before']=before
            result=y04.train_output(a,'O_multi',reader,schedule,context['initial_head_state'],path,device,bool(args.smoke_only),steps)
            t.cuda.synchronize();training_sec=time.perf_counter()-started;peak=t.cuda.max_memory_allocated();after=gpu_endpoint(rt)
            attempt.update(status='TRAINING_RETURNED',cache_seconds=cache_sec,training_seconds=training_sec,peak_allocated_bytes=peak,gpu_endpoints=[before,after],actual_GT_exposures=actual_exposure)
            if args.smoke_only:
                write(marker,dict(status='SMOKE_PASS',seed=args.seed,updates=steps,training_performed=True,formal_checkpoint=False,evaluation_performed=False,cache_seconds=cache_sec,training_seconds=training_sec))
                print('[SMOKE PASS] O_multi 3 updates; no formal checkpoint',flush=True);return 0
            stage='finalize';phase_started=time.perf_counter()
            require(sha(p_cp)==identity['P_checkpoint_sha256'] and sha(p_summary)==identity['P_summary_sha256'],'P provenance changed during training')
            if args.p_budget:require(sha(Path(args.p_budget))==identity['P_budget_sha256'],'P budget changed during training')
            require(sha(snapshot/'model.safetensors')==identity['snapshot_sha256'] and sha(snapshot/'config.json')==identity['snapshot_config_sha256'],'Model snapshot changed during training')
            require(all(sha(pair_paths[s])==pair_shas[s] for s in pair_paths),'Training pair manifest changed during training')
            require(training_inputs(manifests)==fingerprints,'Training input RGB/GT bytes changed during training')
            for rel,expected_sha in identity['source_sha256'].items():require(sha(root/rel)==expected_sha,'Training source changed: '+rel)
            require(sha(__file__)==identity['runner_sha256'] and sha(Path(__file__).with_name('seed_contract.py'))==identity['contract_sha256'],'Training runner changed during training')
            y04.write_checkpoint_provenance(path/'training/O_multi','Y04_O_multi','output_only',list(y04.SEQUENCES),slots,schedule,context['initial_head_hash'],BASE_SHA,len(rows),'Y04_O_multi',pair_paths)
            cp=path/'training/O_multi/Y04_O_multi_final_step1000.pt';summ=path/'training/O_multi/Y04_O_multi_training_summary.json'
            result_obj=t.load(cp,map_location='cpu',weights_only=False)
            require(result_obj.get('initial_adapter_state_sha256')==context['initial_head_hash'] and result_obj.get('sample_order_sha256')==order_sha(args.seed),'O checkpoint actual initialization/order mismatch')
            state=result_obj.get('adapter_state_dict');require(isinstance(state,dict) and state,'O adapter state missing')
            require(all(t.is_tensor(x) and bool(t.isfinite(x).all().item()) for x in state.values()) and sum(x.numel() for x in state.values())==170496,'O adapter tensors invalid or wrong parameter count')
            result_obj.update(y04_outer_seed=args.seed,y04_optimizer_order_seed=args.seed,cvpr_omulti_seed_runner='1.0',cvpr_omulti_seed_runner_sha256=sha(__file__),supervision_sha256=supervision_sha)
            tmp=cp.with_name(cp.name+'.tmp');t.save(result_obj,tmp);os.replace(tmp,cp)
            summary=json.loads(summ.read_text(encoding='utf-8'));summary.update(checkpoint_sha256=sha(cp),y04_outer_seed=args.seed,y04_sample_order_sha256=order_sha(args.seed),supervision_sha256=supervision_sha,actual_GT_exposures=actual_exposure,cache_seconds=cache_sec,current_invocation_training_seconds=training_sec,peak_allocated_bytes=peak)
            write(summ,summary)
            pair=incomplete_pair(args.seed,sha(p_cp),sha(cp),context['initial_head_hash'],supervision_sha,actual_exposure)
            write(path/'matched_pair_evidence_INCOMPLETE.json',pair)
            write(marker,dict(status='TRAINING_ONLY_COMPLETE_INPUT_PROVENANCE_PENDING',seed=args.seed,checkpoint=str(cp),checkpoint_sha256=sha(cp),P_checkpoint_sha256=sha(p_cp),matched_pair_evidence=str(path/'matched_pair_evidence_INCOMPLETE.json'),evaluation_performed=False))
            attempt['status']='COMPLETE';print('[COMPLETE] O_multi seed',args.seed,'checkpoint',cp,flush=True);return 0
    except BaseException as e:
        attempt.update(status='FAILED',stage=stage,phase_elapsed_seconds=time.perf_counter()-phase_started,error=str(e),traceback=traceback.format_exc());write(path/'failure.json',attempt);raise
    finally:
        attempt['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        with attempts.open('a',encoding='utf-8') as f:f.write(json.dumps(attempt)+'\n');f.flush();os.fsync(f.fileno())

def main():
    a=parser().parse_args()
    if a.self_test:self_test();return 0
    return run(a)
if __name__=='__main__':raise SystemExit(main())
