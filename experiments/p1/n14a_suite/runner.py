"""N14-A standalone coverage-first, current-GT-only training and development evaluation."""
from .common import *
import argparse,fcntl,sys,time,traceback,zipfile

def archive(out,y05r=None,coverage_only=False):
    dest=out/('n14a_coverage_review.zip' if coverage_only else 'n14a_review.zip');tmp=dest.with_name(dest.name+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            rel=p.relative_to(out)
            if p.is_file() and p.suffix in ('.csv','.json','.jsonl','.md') and not any(x in ('cache','predictions') for x in rel.parts) and p.name!='failure.json':z.write(p,rel.as_posix())
        for p in sorted(HERE.rglob('*')):
            if p.is_file() and p.suffix in ('.py','.json','.csv') and '__pycache__' not in p.parts:z.write(p,'source/'+p.relative_to(HERE).as_posix())
        if y05r is not None:
            for name in ('protocol.json','summary.json'):z.write(y05r/name,'reference/Y05R_'+name)
    with zipfile.ZipFile(tmp) as z:require(z.testzip() is None,'Review ZIP CRC failed')
    tmp.replace(dest);return dest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--y05r-dir',default=os.environ.get('Y05R_DIR','auto'))
    parser.add_argument('--dataset-protocol',default=os.environ.get('N14A_DATASET_PROTOCOL',os.environ.get('DN_DATASET_PROTOCOL','outputs/y11_confirmation_data/dataset_protocol.json')))
    parser.add_argument('--out-dir',required=False);parser.add_argument('--resume',action='store_true');parser.add_argument('--resume-latest',action='store_true')
    group=parser.add_mutually_exclusive_group();group.add_argument('--preflight-only',action='store_true');group.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--save-depths',action='store_true');a=parser.parse_args()
    verify_hashes({str(HERE/k):v for k,v in read(HERE/'installed_manifest.json').items()});policy=read(HERE/'policy.json');identity=read(HERE/'identity.json')
    if a.resume_latest:
        require(not a.out_dir and not a.preflight_only and not a.smoke_only,'Use --resume-latest alone');paths=[]
        for p in (ROOT/'outputs').glob('gt4a2_n14a_*/validation/protocol.json'):
            x=read(p)
            if x.get('experiment')=='N14-A' and not x['partial_smoke']:paths.append(p)
        require(bool(paths),'No formal N14-A checkpoint; if stopped during coverage audit, rerun without --resume-latest')
        a.out_dir=str(max(paths,key=lambda p:p.stat().st_mtime_ns).parent);a.resume=True
    require(a.out_dir,'Use run_n14a.sh, or provide --out-dir');out=Path(a.out_dir).resolve()
    require(out.is_relative_to(ROOT/'outputs') and out.parent.name.startswith('gt4a2_n14a_'),'Use isolated outputs/gt4a2_n14a_<timestamp>/validation')
    if a.resume:
        require(not a.preflight_only and not a.smoke_only,'Resume formal runs only');frozen=read(out/'protocol.json')
        require(frozen['experiment']=='N14-A' and not frozen['partial_smoke'],'Not a formal N14-A protocol')
        a.y05r_dir=frozen['y05r_dir'];a.dataset_protocol=frozen['dataset_protocol'];a.save_depths=frozen['save_depths']
    out.mkdir(parents=True,exist_ok=True);phase_name='read-only-source-and-coverage';start=time.perf_counter()
    with (out/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            audit=source_audit(a,out,policy,identity);y05r,prior,cases,coverage,source=audit
            for seq in policy['old_sequences']:print('[GT COVERAGE]',seq,','.join(f'h{r["phase"]}={r["matched_GT"]}' for r in coverage if r['sequence']==seq),flush=True)
            if not source['coverage_passed']:
                summary=dict(status='COVERAGE_BLOCKED',experiment='N14-A',training_performed=False,CUDA_initialized=False,minimum=policy['minimum_GT_per_sequence_phase'],failed_cells=[r for r in coverage if not r['passed']],method_pass=False)
                write(out/'summary.json',summary);(out/'REPORT.md').write_text('# N14-A coverage blocked\n\n每个训练序列 × h1..7 至少需要 8 个 matched-GT 帧。未训练、未初始化 CUDA；不可插值/复用 GT 或自动扩展前缀。\n')
                print('[COVERAGE BLOCKED]',archive(out,y05r,True),flush=True);return 2
            print('[COVERAGE PASS] 21 strata; current-frame samples=',len(cases),flush=True)
            if a.preflight_only:print('[READ-ONLY PREFLIGHT COMPLETE] No Torch/CUDA, cache or optimizer; inspect data_coverage.csv',flush=True);return 0
            uuid,gpu_path=gpu_lock_path()
            with gpu_path.open('a+') as gpu:
                print('[GPU QUEUE] waiting before Torch/CUDA initialization',flush=True);fcntl.flock(gpu,fcntl.LOCK_EX);phase_name='runtime-preflight'
                from .runtime import prepare
                ctx=prepare(a,out,audit);actual=str(getattr(ctx.rt.torch.cuda.get_device_properties(0),'uuid','')).removeprefix('GPU-');require(actual==uuid,'CUDA device differs from locked GPU')
                protocol=out/'protocol.json'
                if protocol.exists():require(a.resume and read(protocol)==ctx.plan,'Frozen protocol differs; do not edit old protocol/hash files')
                else:require(not a.resume,'No formal protocol to resume');write(protocol,ctx.plan)
                ctx.signature=sha(protocol)
                print('[PREFLIGHT PASS] N14-A; 9244 params/arm; A0/A1 current GT only; NO state writeback; NO future GT; NO DOWNLOAD',flush=True)
                from .training import train_all
                phase_name='smoke-cache-gradient-training';train_all(ctx,a.resume)
                from .state_audit import audit_all
                phase_name='state-invariance';audit_all(ctx,a.smoke_only)
                from .evaluation import evaluate_all
                phase_name='quality-timing-report';evaluate_all(ctx,a.save_depths,a.smoke_only)
                phase_name='final-verification';verify_hashes(ctx.plan['input_sha256']);(out/'failure.json').unlink(missing_ok=True)
                if a.smoke_only:print('[SMOKE COMPLETE] Real model checks and two updates/arm; not a method result',flush=True);return 0
                package=archive(out,y05r)
                write(out/'completed.json',dict(signature=ctx.signature,review_zip_sha256=sha(package),current_invocation_wall_sec=time.perf_counter()-start))
                print('[COMPLETE]',package,flush=True);print('[N14-B] not started automatically',flush=True);return 0
        except BaseException as e:
            write(out/'failure.json',dict(phase=phase_name,error=str(e),traceback=traceback.format_exc(),current_invocation_wall_sec=time.perf_counter()-start));print('[STOPPED]',phase_name,'see',out/'failure.json',flush=True);raise

if __name__=='__main__':raise SystemExit(main())
