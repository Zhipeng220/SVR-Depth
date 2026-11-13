#!/usr/bin/env python3
"""Development-only Y09 stream canary: A, P, O_multi; K16/32/64; crowd2 prefix."""
import csv,hashlib,json,os,sys,time,traceback
from pathlib import Path
from stream_factory import build_stream, verify_counts

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    root=Path.cwd().resolve(); suite=root/'experiments/p1'; sys.path[:0]=[str(root),str(suite)]
    pin={'y08y09_models.py':'3b5e24786ab61c20c5b780e01cef2006a1a7868fa88b071ea63588d696d0e5b8',
         'cvpr_m05_formal.py':'efea979881a187532e6e00cc24ef265427038c73ffabfe8763bf5e6a74a18826',
         'gt4a2_y04a_native_pre_readout_eval.py':'6fbe1a2464a7f9756f9ea68cf05f3476a89b005b8e6f889dd1a0093f231c3116',
         'gt4a2_y11_fallback.py':'e4c5d46cbe48b56c1631090959803f9d8db4690e6426aa60f8ff68318e0503ce'}
    for name,digest in pin.items():
        if not (suite/name).is_file() or sha(suite/name)!=digest: raise RuntimeError('Reviewed source missing/changed: '+name)
    import cvpr_m05_formal as m05
    from gt4a2_y04a_native_pre_readout_eval import load_runtime
    from gt4a2_y11_fallback import gpu_lock_path
    import fcntl
    snapshot=Path(os.environ.get('VDEEP_SNAPSHOT','/missing')).resolve()
    pcp=root/m05.KNOWN_P_CHECKPOINT; ocp=root/'outputs/gt4a2_y04_validation_fix3_20261025_225902/training/O_multi/Y04_O_multi_final_step1000.pt'
    pairs=root/'outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv'
    m05.verify_sources()
    for path,digest in ((snapshot/'model.safetensors',m05.EXPECTED_BASE_SHA256),(pcp,m05.KNOWN_P_CHECKPOINT_SHA256),(ocp,'558e1850d7b08939bef0db0eb4f7a48fdff8a2fb58417cd4c380ee3c6a6f6948')):
        if not path.is_file() or sha(path)!=digest:raise RuntimeError('Reviewed asset missing or changed: '+str(path))
    out=root/'outputs'/('cvpr_m01_m02_dev_smoke_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()));out.mkdir(parents=True,exist_ok=False)
    try:
        uuid,path=gpu_lock_path()
        with path.open('a+') as lock:
            print('[GPU QUEUE] waiting for shared GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            rt=load_runtime();t=rt.torch;np=rt.np
            if not t.cuda.is_available() or str(getattr(t.cuda.get_device_properties(0),'uuid','')).removeprefix('GPU-')!=uuid:raise RuntimeError('CUDA device unavailable or mismatched GPU lock')
            t.set_num_threads(4)
            from y08y09_models import Stream,GainHead
            from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead
            from y11_common import metric
            ps=t.load(pcp,map_location='cpu',weights_only=False)['repair_head_state_dict']
            os_=t.load(ocp,map_location='cpu',weights_only=False)['adapter_state_dict']
            records=rt.exp1.read_manifest(pairs,training=False)[:40]
            if len(records)!=40 or [int(r['frame_index_zero_based']) for r in records]!=list(range(40)):raise RuntimeError('Need continuous crowd2 RGB0..39')
            result=[];frames=[]
            for k in (16,32,64):
                for method in ('A','O_multi','P'):
                    model=rt.gt2.load_model(snapshot,'pre_readout' if method=='P' else 'off',t.device('cuda')); adapter=None
                    if method=='P':
                        head=model.pixel_decoder.persistent_feature_repair;head.load_state_dict(ps,strict=True);model.pixel_decoder.persistent_feature_repair=GainHead(head,1.)
                    if method=='O_multi':
                        adapter=PersistentFeatureRepairHead(channels=512,bottleneck=32,levels=4).cuda();adapter.load_state_dict(os_,strict=True);adapter.eval()
                    model.eval()
                    for p in model.parameters():p.requires_grad_(False)
                    shape=rt.exp1.shape_info_for(model,Path(records[0]['rgb_path']),t.device('cuda'))
                    stream=build_stream(Stream,rt,model,shape,method,k,adapter)
                    scores=[]
                    try:
                        for i,record in enumerate(records):
                            with t.no_grad(): pred,meta=stream.step(i,record)
                            if meta['keyframe']!=(i%k==0) or meta['age']!=i%k:raise RuntimeError('Fixed schedule differs')
                            if not np.isfinite(pred).all() or not (pred>0).all():raise RuntimeError('Invalid prediction')
                            row=dict(method=method,K=k,index=i,matched=str(record['gt_match_status']).upper()=='MATCHED',AbsRel='',RMSE='',delta1='')
                            if row['matched']:
                                gt=rt.exp1.load_frame(record,shape,t.device('cuda'),need_gt=True)['gt_np']; mm=metric(np,pred,gt)
                                row.update({q:float(mm[q]) for q in ('AbsRel','RMSE','delta1')});scores.append(row)
                            frames.append(row)
                        if not scores:raise RuntimeError('No matched GT in development prefix')
                        verify_counts(stream.counts,method,k,len(records))
                        result.append(dict(method=method,K=k,frames=40,matched_frames=len(scores),counts=stream.counts,**{q:sum(r[q] for r in scores)/len(scores) for q in ('AbsRel','RMSE','delta1')}))
                        print('[DEV SMOKE]',method,'K'+str(k),'40 frames',flush=True)
                    finally:
                        del stream,model,adapter;t.cuda.empty_cache()
            with (out/'frames.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(frames[0]));w.writeheader();w.writerows(frames)
            summary=dict(status='DEV_SMOKE_COMPLETE',development_only=True,independent_test=False,training_performed=False,sequence='crowd2_prefix',frames=40,methods=['A','O_multi','P'],missing=['O_fast','additional seeds'],scope='Single existing checkpoint; K16/32/64 development API smoke; no formal conclusions or timing.',results=result,snapshot_sha256=sha(snapshot/'model.safetensors'),P_sha256=sha(pcp),O_multi_sha256=sha(ocp),manifest_sha256=sha(pairs))
            (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('[DEV SMOKE] summary:',out/'summary.json',flush=True)
            return 0
    except BaseException as e:
        (out/'failure.json').write_text(json.dumps(dict(error=str(e),traceback=traceback.format_exc()),indent=2),encoding='utf-8');raise

if __name__=='__main__': raise SystemExit(main())
