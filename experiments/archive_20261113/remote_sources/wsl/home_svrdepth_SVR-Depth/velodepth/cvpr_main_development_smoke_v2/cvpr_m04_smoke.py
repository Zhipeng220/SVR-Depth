#!/usr/bin/env python3
"""M04 DEVELOPMENT ONLY: 40 crowd2 prefix frames, single reviewed Y04 P seed."""
import argparse, csv, hashlib, json, os, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
from m04_branch import ARMS,branch_step

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT','')); args=parser.parse_args()
    root=Path.cwd().resolve(); suite=root/'experiments'/'p1'; sys.path[:0]=[str(root),str(suite)]
    pinned={'y08y09_models.py':'3b5e24786ab61c20c5b780e01cef2006a1a7868fa88b071ea63588d696d0e5b8',
            'y10_runtime.py':'4c3802822dd1eafb22ed1288b8f0494387f113822c305b9598054cbdaa545ec2',
            'y10_core.py':'220702c9198a90586a4f58a0a9d1ea374131421f57b943861289eca5096b1b4f',
            'cvpr_m05_formal.py':'efea979881a187532e6e00cc24ef265427038c73ffabfe8763bf5e6a74a18826',
            'gt4a2_y04a_native_pre_readout_eval.py':'6fbe1a2464a7f9756f9ea68cf05f3476a89b005b8e6f889dd1a0093f231c3116',
            'gt4a2_y11_fallback.py':'e4c5d46cbe48b56c1631090959803f9d8db4690e6426aa60f8ff68318e0503ce'}
    for name,digest in pinned.items():
        if not (suite/name).is_file() or sha(suite/name)!=digest: raise RuntimeError('Reviewed source missing/changed: '+name)
    import cvpr_m05_formal as m05
    from gt4a2_y04a_native_pre_readout_eval import load_runtime
    from gt4a2_y11_fallback import gpu_lock_path
    from y10_runtime import Engine,clone_state,equal_state
    import fcntl
    snapshot=Path(args.snapshot).resolve(); cp=root/m05.KNOWN_P_CHECKPOINT
    pairs=root/'outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv'
    if not args.snapshot or not (snapshot/'config.json').is_file() or not (snapshot/'model.safetensors').is_file(): raise RuntimeError('Set VDEEP_SNAPSHOT to reviewed model snapshot')
    m05.verify_sources()
    if sha(snapshot/'model.safetensors')!=m05.EXPECTED_BASE_SHA256 or sha(cp)!=m05.KNOWN_P_CHECKPOINT_SHA256: raise RuntimeError('Reviewed snapshot or P checkpoint hash differs')
    out=root/'outputs'/('cvpr_m04_dev_smoke_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())); out.mkdir(parents=True,exist_ok=False)
    try:
        uuid,lock_path=gpu_lock_path()
        with lock_path.open('a+') as lock:
            print('[GPU QUEUE] waiting for shared GPU lock',flush=True); fcntl.flock(lock,fcntl.LOCK_EX)
            rt=load_runtime(); torch=rt.torch; np=rt.np
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
            actual=str(getattr(torch.cuda.get_device_properties(0),'uuid','')).removeprefix('GPU-')
            if actual!=uuid: raise RuntimeError('Selected CUDA device differs from GPU lock')
            torch.set_num_threads(4)
            obj=torch.load(cp,map_location='cpu',weights_only=False); head=obj['repair_head_state_dict']
            records=rt.exp1.read_manifest(pairs,training=False)[:40]
            if len(records)!=40 or [int(x['frame_index_zero_based']) for x in records]!=list(range(40)): raise RuntimeError('Need continuous crowd2 RGB0..39')
            ctx=SimpleNamespace(rt=rt,args=SimpleNamespace(snapshot=str(snapshot)),head_state=head,manifests={'crowd2':records})
            predictions={}; counts={}; state_checks=[]
            for arm in ARMS:
                eng=Engine(ctx,'crowd2'); state=prev=None; preds=[]; fast=0
                try:
                    for i in range(40):
                        frame=eng.load(i,False)
                        static=eng.static(prev,frame,'P') if i%16 else None
                        def step(repeats):
                            s,d,_,_=eng.step(state,prev,frame,16,'P',repeats=repeats,static=static)
                            return s,d
                        state,depth,n=branch_step(arm,i,16,step,clone_state); fast+=n
                        if not np.isfinite(depth).all() or not (depth>0).all(): raise RuntimeError(f'Invalid prediction: {arm}/{i}')
                        preds.append(depth.copy()); prev=frame
                    predictions[arm]=preds; counts[arm]=fast
                finally: eng.close()
                print('[DEV SMOKE]',arm,'frames=40 physical_fast_steps=',fast,flush=True)
            # OFF and OUTPUT_ONLY have identical persisted native state; outputs can differ.
            # Replay against Y08/Y09's actual native P stream, not an inferred reference.
            from y08y09_models import Stream,GainHead
            model=rt.gt2.load_model(snapshot,'pre_readout',torch.device('cuda'))
            head_module=model.pixel_decoder.persistent_feature_repair; head_module.load_state_dict(head,strict=True)
            model.pixel_decoder.persistent_feature_repair=GainHead(head_module,1.)
            model.eval()
            for p in model.parameters(): p.requires_grad_(False)
            shape=rt.exp1.shape_info_for(model,Path(records[0]['rgb_path']),torch.device('cuda'))
            stream=Stream(rt,model,dict(family='P',schedule='K16',gain=1.,frequency=1),shape)
            errors=[]
            try:
                for i,r in enumerate(records):
                    with torch.no_grad(): ref,_=stream.step(i,r)
                    errors.append(float(np.max(np.abs(ref-predictions['P_NATIVE'][i]))))
            finally:
                del stream,model; torch.cuda.empty_cache()
            # Exact same-parent comparisons are checked with one fresh Engine on the first two frames.
            eng=Engine(ctx,'crowd2')
            try:
                f0=eng.load(0,False); s0,_,_,_=eng.step(None,None,f0,16,'P',repeats=0)
                f1=eng.load(1,False); static=eng.static(f0,f1,'P')
                def probe(repeats):
                    s,d,_,_=eng.step(s0,f0,f1,16,'P',repeats=repeats,static=static); return s,d
                checks={arm:branch_step(arm,1,16,probe,clone_state) for arm in ARMS}
                off,o,f,p=(checks[a] for a in ARMS)
                state_checks=[equal_state(torch,off[0],o[0]),np.array_equal(off[1],f[1]),np.array_equal(o[1],p[1]),
                    all(torch.equal(x,y) for x,y in zip(f[0][2],p[0][2]))]
            finally: eng.close()
            passed=all(state_checks) and max(errors)<=1e-4
            summary=dict(status='DEV_SMOKE_PASS' if passed else 'DEV_SMOKE_FAILED',development_only=True,test_set_accessed=False,training_performed=False,frames=40,sequence='crowd2_prefix',K=16,seed_count=1,
                p_native_vs_y09_max_abs=max(errors),p_native_tolerance=1e-4,same_parent_checks=state_checks,physical_fast_steps=counts,
                interpretation='Diagnostics use two complete fast steps; cost is not an optimized M04 timing. This is not formal M04 or independent-test evidence.',
                snapshot_sha256=sha(snapshot/'model.safetensors'),checkpoint_sha256=sha(cp),manifest_sha256=sha(pairs),gpu=torch.cuda.get_device_name(0))
            (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            print('[DEV SMOKE]',summary['status'],'summary:',out/'summary.json',flush=True)
            return 0 if passed else 2
    except BaseException as e:
        (out/'failure.json').write_text(json.dumps(dict(error=str(e),traceback=traceback.format_exc()),indent=2),encoding='utf-8'); raise

if __name__=='__main__': raise SystemExit(main())
