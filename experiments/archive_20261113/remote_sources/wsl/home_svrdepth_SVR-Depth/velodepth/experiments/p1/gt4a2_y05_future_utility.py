#!/usr/bin/env python3
"""Y05-H4 entry point: new training, not the earlier frozen-head diagnostic."""
from y05_common import *
import argparse,fcntl,time,traceback

def parser():
    p=argparse.ArgumentParser(description='Y05-H4 single-pulse future-utility training and development suite')
    p.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT',''))
    p.add_argument('--y04-dir',default=os.environ.get('Y04_DIR','outputs/gt4a2_y04_validation_fix3_20261025_225902'))
    for name in ('y09','y03','y10'):p.add_argument('--'+name+'-dir',default=os.environ.get(name.upper()+'_DIR','auto'))
    for s in SEQS:p.add_argument('--'+s+'-pair-csv',default=f'outputs/gt4a2_exp1/gt4a2_exp1_{s}_pairs_used.csv')
    p.add_argument('--out-dir',default='outputs/gt4a2_y05_manual/validation');p.add_argument('--resume',action='store_true')
    p.add_argument('--stage',choices=('all','train','evaluate','timing','report'),default='all')
    g=p.add_mutually_exclusive_group();g.add_argument('--plan-only',action='store_true');g.add_argument('--preflight-only',action='store_true');g.add_argument('--smoke-only',action='store_true')
    return p

def verify_complete(out,signature,smoke):
    summary=jread(out/'summary.json')
    require(summary['signature']==signature and summary['status']==('SMOKE_PASS' if smoke else 'Y05_DEVELOPMENT_SUITE_COMPLETE'),'Incomplete Y05 summary')
    require(summary['partial_smoke']==smoke and not summary['Y11_pass'] and summary['no_automatic_method_pass'],'Completion claims differ')
    with zipfile.ZipFile(out/'y05_review.zip') as z:require(z.testzip() is None and json.loads(z.read('summary.json'))==summary,'Review archive mismatch')

def main(argv=None):
    args=parser().parse_args(argv)
    if args.plan_only:
        from y05_eval import configurations
        print(json.dumps(dict(settings=frozen_settings(),prefix_candidates=prefix_samples(),probe_cases=probe_cases(),configurations=configurations(),pairing_status_filter='actual prefix samples require current GT and >=1 future GT; no error-based selection'),ensure_ascii=False,indent=2));return 0
    out=Path(args.out_dir).resolve()
    require(out.parent.name.startswith('gt4a2_y05_'),'Use own outputs/gt4a2_y05_.../{validation,smoke,preflight} namespace')
    require(not(args.resume and args.smoke_only),'Smoke and formal recovery are separate')
    if args.resume:
        require((out/'protocol.json').is_file(),'No formal protocol; if stopped before formal, rerun launcher without --resume')
        frozen=jread(out/'protocol.json');require(frozen['experiment']=='Y05' and not frozen['partial_smoke'],'Foreign/smoke checkpoint')
        for field,path in frozen['paths'].items():
            flag='--'+field.replace('_','-')
            if argv is None and flag in sys.argv:require(str(Path(getattr(args,field)).resolve())==path,'Resume source path must remain frozen')
            setattr(args,field,path)
    require(bool(args.snapshot),'Set VDEEP_SNAPSHOT')
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();completed=False;phase='preflight';journaled=False
    with (out/'.run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This exact output task is already running')
        try:
            ctx=preflight(args);ctx.out=out
            for field in ('y04_dir','y09_dir','y03_dir','y10_dir'):
                prior=Path(ctx.plan['paths'][field]);require(prior not in (out,*out.parents),'Output cannot enter old results')
            protocol=out/'protocol.json'
            if protocol.exists():require(jread(protocol)==ctx.plan,'Source/data/environment/plan changed; do not edit protocol.json')
            else:jwrite(protocol,ctx.plan)
            ctx.signature=sha(protocol)
            print(f'[PREFLIGHT PASS] {VERSION}: {len(ctx.plan["sample_plan"])} prefix samples, H4, warmstart P, new training',flush=True)
            if args.preflight_only:completed=True;return 0
            if (out/'summary.json').exists() and (out/'y05_review.zip').exists():
                verify_complete(out,ctx.signature,args.smoke_only);completed=True;print('[ALREADY COMPLETE]',out/'y05_review.zip');return 0
            torch=ctx.rt.torch;torch.set_num_threads(4)
            from y05_train import seed,train_all
            from y05_eval import run_quality,run_probes,run_timing,report
            seed(ctx);uuid=ctx.plan['environment']['uuid'] or ctx.plan['environment']['gpu_name']
            gpu=Path('/tmp')/('svrdepth_diagnostics_'+hashlib.sha256(uuid.encode()).hexdigest()[:16]+'.lock')
            print('[GPU QUEUE] waiting for exclusive access',flush=True)
            with gpu.open('a+') as gpulock:
                fcntl.flock(gpulock,fcntl.LOCK_EX)
                if args.stage in ('all','train'):phase='training';train_all(ctx)
                if args.stage in ('all','evaluate'):
                    phase='full-RGB evaluation';run_quality(ctx)
                    phase='single-pulse future probes';run_probes(ctx)
                if args.stage in ('all','timing'):phase='timing';run_timing(ctx)
                if args.stage in ('all','report'):
                    phase='report';report(ctx)
                    for input_path,digest in ctx.plan['input_sha256'].items():require(sha(input_path)==digest,'Input changed during Y05: '+input_path)
                    with (out/'invocations.jsonl').open('a') as f:f.write(json.dumps(dict(stage=args.stage,phase=phase,completed=True,resume=args.resume,physical_invocation_wall_sec=time.perf_counter()-start,unix_time=time.time(),scope='through input verification; ZIP finalization follows'))+'\n')
                    journaled=True
                    archive(ctx.out);verify_complete(out,ctx.signature,args.smoke_only)
                    print('[VERIFIED]',out/'y05_review.zip',flush=True)
                else:print(f'[STAGE COMPLETE] {args.stage}; continue: bash run_y05.sh --resume {out}',flush=True)
            (out/'failure.json').unlink(missing_ok=True);completed=True;return 0
        except BaseException as exc:
            jwrite(out/'failure.json',dict(phase=phase,error=str(exc),traceback=traceback.format_exc(),resume_command=f'bash run_y05.sh --resume {out}' if (out/'protocol.json').is_file() else 'bash run_y05.sh'))
            # Do not leave a successful summary after a failed report/input verification.
            if phase=='report':(out/'summary.json').unlink(missing_ok=True)
            raise
        finally:
            if not journaled or not completed:
                with (out/'invocations.jsonl').open('a') as f:f.write(json.dumps(dict(stage=args.stage,phase=phase,completed=completed,resume=args.resume,physical_invocation_wall_sec=time.perf_counter()-start,unix_time=time.time()))+'\n')

if __name__=='__main__':raise SystemExit(main())
