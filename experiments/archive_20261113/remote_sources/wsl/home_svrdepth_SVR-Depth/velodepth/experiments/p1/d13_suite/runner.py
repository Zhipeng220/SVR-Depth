"""Formal two-arm D13/N13 training, natural-history development evaluation and recovery."""
from .common import *
import argparse,fcntl,sys,time,traceback,zipfile

def archive(ctx):
    name=ctx.policy['experiment'].lower()+'_review.zip';dest=ctx.out/name;q=dest.with_name(dest.name+'.tmp')
    with zipfile.ZipFile(q,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(ctx.out.rglob('*')):
            rel=p.relative_to(ctx.out)
            if p.is_file() and p.suffix in ('.json','.csv','.md','.jsonl') and not any(x in ('cache','predictions') for x in rel.parts) and p.name!='failure.json':z.write(p,rel.as_posix())
        for p in sorted(HERE.iterdir()):
            if p.suffix in ('.py','.json'):z.write(p,'source/'+p.name)
        z.write(ctx.y05r/'protocol.json','reference/Y05R_protocol.json')
        z.write(ctx.y05r/'summary.json','reference/Y05R_summary.json')
    with zipfile.ZipFile(q) as z:require(z.testzip() is None,'Review ZIP CRC failed')
    q.replace(dest);return dest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--y05r-dir',default=os.environ.get('Y05R_DIR','auto'))
    p.add_argument('--dataset-protocol',default=os.environ.get('DN_DATASET_PROTOCOL','outputs/y11_confirmation_data/dataset_protocol.json'))
    p.add_argument('--out-dir');p.add_argument('--resume',action='store_true');p.add_argument('--resume-latest',action='store_true')
    group=p.add_mutually_exclusive_group();group.add_argument('--smoke-only',action='store_true');group.add_argument('--preflight-only',action='store_true')
    p.add_argument('--save-depths',action='store_true',help='Keep all scored float32 depths; default keeps per-frame hashes plus fixed numerical probes')
    a=p.parse_args();policy=read(HERE/'policy.json');series=policy['experiment'].lower()
    # Every installed file is verified before importing Torch or initializing CUDA.
    verify_hashes({str(HERE/k):v for k,v in read(HERE/'installed_manifest.json').items()})
    if a.resume_latest:
        require(not a.out_dir and not a.smoke_only and not a.preflight_only,'Use --resume-latest alone')
        candidates=[]
        for f in (ROOT/'outputs').glob(f'gt4a2_{series}_*/validation/protocol.json'):
            x=read(f)
            if x.get('experiment')==policy['experiment'] and not x.get('partial_smoke'):candidates.append(f)
        require(bool(candidates),'No formal checkpoint; if interrupted during smoke/preflight, rerun launcher without --resume-latest')
        a.out_dir=str(max(candidates,key=lambda f:f.stat().st_mtime_ns).parent);a.resume=True
    require(a.out_dir,'--out-dir is required (use run_'+series+'.sh)')
    out=Path(a.out_dir).resolve();require(out.is_relative_to(ROOT/'outputs') and out.parent.name.startswith('gt4a2_'+series+'_'),'Use isolated outputs/gt4a2_'+series+'_<timestamp>/validation')
    if a.resume:
        require(not a.smoke_only and not a.preflight_only,'Only formal training can resume')
        saved=read(out/'protocol.json');require(saved['experiment']==policy['experiment'] and not saved['partial_smoke'],'Not a formal checkpoint of this series')
        a.y05r_dir=saved['y05r_dir'];a.dataset_protocol=saved['dataset_protocol'];a.save_depths=saved['save_depths']
    out.mkdir(parents=True,exist_ok=True);phase='source-preflight';start=time.perf_counter()
    with (out/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        uuid,gpu_path=gpu_lock_path()
        with gpu_path.open('a+') as gpu:
            print('[GPU QUEUE] waiting before Torch/CUDA initialization',flush=True);fcntl.flock(gpu,fcntl.LOCK_EX)
            try:
                from .runtime import prepare
                ctx=prepare(a,out);ctx.plan['save_depths']=a.save_depths
                ctx.plan['prediction_retention']='all scored float32 compressed depths plus probes' if a.save_depths else 'every RGB prediction SHA256 and per-frame metrics; five fixed float32 numerical probes/video/method; full dense prediction retention disabled'
                actual=str(getattr(ctx.rt.torch.cuda.get_device_properties(0),'uuid','')).removeprefix('GPU-')
                require(actual==uuid,'Locked GPU differs from CUDA device')
                f=out/'protocol.json'
                if f.exists():require(a.resume and read(f)==ctx.plan,'Protocol changed; never edit old protocol or source hashes to force resume')
                else:require(not a.resume,'No frozen protocol for resume');write(f,ctx.plan)
                ctx.signature=sha(f)
                print('[PREFLIGHT PASS]',policy['experiment'],policy['method'],'two arms;',ctx.plan['updates'],'updates/arm; H1; NO DATA DOWNLOAD',flush=True)
                print('[PROTOCOL]',f,flush=True)
                if a.preflight_only:return 0
                from .training import train_all
                phase='training';train_all(ctx,a.resume)
                from .evaluation import evaluate_all
                phase='evaluation';evaluate_all(ctx,a.save_depths,a.smoke_only)
                phase='final-verification';verify_hashes(ctx.plan['input_sha256'])
                if a.smoke_only:print('[SMOKE PASS] actual VeloDepth training/gradients/quality/timing; no method claim',flush=True);return 0
                phase='archive';result=archive(ctx)
                print('[COMPLETE]',result,flush=True);print('[METHOD PASS] False; read screening evidence in REPORT.md',flush=True)
                write(out/'completed.json',dict(signature=ctx.signature,review_zip_sha256=sha(result),current_invocation_wall_sec=time.perf_counter()-start))
                (out/'failure.json').unlink(missing_ok=True);return 0
            except BaseException as e:
                write(out/'failure.json',dict(phase=phase,error=str(e),traceback=traceback.format_exc()))
                print('[STOPPED]',phase,'see',out/'failure.json',flush=True);raise

if __name__=='__main__':raise SystemExit(main())
