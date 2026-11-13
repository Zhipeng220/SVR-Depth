#!/usr/bin/env python3
"""Three rotated, synchronized timing trials for frozen Bonn quality trajectories."""
import argparse,csv,os,statistics,sys,time,traceback
from pathlib import Path
from contract import digest,read_json,require,verify_counts
from quality import Predictor,verify,write_json,environment
from report import q

ROOT=Path.cwd().resolve()

def write_frames(path,rows):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['index','keyframe','age','latency_ms']);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def run(plan_path,quality_dir,out):
    plan_path=Path(plan_path).resolve();plan=read_json(plan_path);quality_dir=Path(quality_dir).resolve()
    previous=read_json(quality_dir/'summary.json');require(previous['plan_sha256']==digest(plan_path) and previous['status']=='QUALITY_ONLY_COMPLETE','Need matching completed quality pass')
    require(previous['m04_native_replay_sha256']==digest(quality_dir/'m04_native_replay.json'),'M04 native replay audit changed')
    out=Path(out).resolve();require(not out.exists(),'Timing output must be new');out.mkdir(parents=True)
    failures=[];completed=[];completion_hashes={}
    try:
        verify(plan);sys.path[:0]=[str(ROOT),str(ROOT/'experiments/p1')]
        from gt4a2_y11_fallback import gpu_lock_path
        from gt4a2_y04a_native_pre_readout_eval import load_runtime
        import fcntl
        uuid,lockpath=gpu_lock_path()
        with lockpath.open('a+') as lock:
            print('[GPU QUEUE] waiting for shared GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            rt=load_runtime();t=rt.torch;np=rt.np
            require(t.cuda.is_available() and str(getattr(t.cuda.get_device_properties(0),'uuid','')).removeprefix('GPU-')==uuid,'CUDA device differs from lock')
            from gt4a2_y08cal_y09pareto import gpu_endpoint
            t.set_num_threads(4);runtime_environment=environment(rt);require(runtime_environment==previous['environment'],'Runtime environment differs from quality pass');units=plan['units']
            for repeat in range(1,4):
                for seq_no,s in enumerate(plan['sequences']):
                    records=rt.exp1.read_manifest(Path(s['manifest']),training=False);n=len(records);require(n==s['RGB'],'Manifest count changed')
                    shift=(repeat-1+seq_no)%len(units);ordered=units[shift:]+units[:shift]
                    for position,unit in enumerate(ordered):
                        key=s['sequence']+'/'+unit['id']+f'/r{repeat}';folder=out/s['sequence']/unit['id'];folder.mkdir(parents=True,exist_ok=True)
                        frame_times=[]
                        try:
                            meta=read_json(quality_dir/'quality'/s['sequence']/unit['id']/'complete.json')
                            require(previous['completion_hashes'].get(s['sequence']+'/'+unit['id'])==digest(quality_dir/'quality'/s['sequence']/unit['id']/'complete.json'),'Quality completion metadata changed')
                            require(meta['failed']==0,'Timing rejected: quality unit contains invalid or unavailable scored predictions')
                            require(meta['frames_sha256']==digest(quality_dir/'quality'/s['sequence']/unit['id']/'frames.csv'),'Quality table changed')
                            predictor=Predictor(rt,plan,unit,records)
                            try:
                                for r in records:Path(r['rgb_path']).read_bytes()
                                for i,r in enumerate(records[:min(n,32)]):predictor.step(i,r)
                                t.cuda.synchronize();predictor.reset();before=gpu_endpoint(rt);require(before.get('clean') is True,'GPU endpoint contaminated before timing')
                                t.cuda.reset_peak_memory_stats();times=[];frame_times=[];probes=0
                                for i,r in enumerate(records):
                                    t.cuda.synchronize();start=time.perf_counter();pred,meta_step=predictor.step(i,r);t.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
                                    frame_times.append(dict(index=i,keyframe=meta_step['keyframe'],age=meta_step['age'],latency_ms=times[-1]))
                                    require(meta_step['keyframe']==(i%unit['k']==0) and int(meta_step['age'])==i%unit['k'],'Timing schedule differs')
                                    if i in (0,n-1) or i%unit['k'] in (1,unit['k']-1):
                                        rel=f"quality/{s['sequence']}/{unit['id']}/depth/{i:06d}.npz";p=quality_dir/rel
                                        require(meta['prediction_sha256'].get(rel)==digest(p),'Quality prediction file changed')
                                        with np.load(p,allow_pickle=False) as z:expected=z['depth']
                                        require(np.allclose(pred,expected,rtol=1e-5,atol=1e-4,equal_nan=False),'Timing trajectory differs from quality');probes+=1
                                    if i%200==0 or i+1==n:print('[TIMING]',key,i+1,'/',n,flush=True)
                                verify_counts(unit,n,predictor.counts());after=gpu_endpoint(rt);clean=bool(before.get('clean') and after.get('clean'))
                                obj=dict(sequence=s['sequence'],unit=unit['id'],repeat=repeat,position=position,frames=n,mean_ms=statistics.mean(times),p50_ms=q(times,.5),p95_ms=q(times,.95),throughput_fps=1000/statistics.mean(times),peak_allocated_bytes=t.cuda.max_memory_allocated(),endpoints=[before,after],endpoint_clean=clean,quality_probes=probes,counts=predictor.counts(),
                                    scope='warm file cache; RGB read/decode/preprocess, DIS, full model/head/readout, CPU depth and CUDA synchronization; no GT, metrics, writes or model loading')
                                if not clean:
                                    frame_path=folder/f'r{repeat}.rejected.frames.csv';write_frames(frame_path,frame_times);obj['frames_sha256']=digest(frame_path)
                                    write_json(folder/f'r{repeat}.rejected.json',obj);raise RuntimeError('GPU endpoint contaminated after timing; trial rejected')
                                require(probes>0,'No quality replay probes');frame_path=folder/f'r{repeat}.frames.csv';write_frames(frame_path,frame_times);obj['frames_sha256']=digest(frame_path)
                                write_json(folder/f'r{repeat}.json',obj);completed.append(key);completion_hashes[key]=digest(folder/f'r{repeat}.json')
                            finally:predictor.close()
                        except Exception as e:
                            failure=dict(unit=key,error=str(e),traceback=traceback.format_exc(),completed_frames=len(frame_times))
                            if frame_times:
                                try:
                                    partial=folder/f'r{repeat}.failure.frames.csv';write_frames(partial,frame_times);failure['partial_frames_sha256']=digest(partial)
                                except Exception as save_error:failure['partial_save_error']=str(save_error)
                            write_json(folder/f'r{repeat}.failure.json',failure);failures.append(failure);print('[TIMING FAILED]',key,str(e),flush=True)
            verify(plan)
        status='TIMING_COMPLETE' if not failures else 'TIMING_INCOMPLETE'
        write_json(out/'summary.json',dict(status=status,plan_sha256=digest(plan_path),quality_summary_sha256=digest(quality_dir/'summary.json'),completed=completed,completion_hashes=completion_hashes,failures=[dict(unit=x['unit'],error=x['error']) for x in failures],
            repeats=3,rotation='(repeat-1 + sequence index) mod unit count',formal_v21_complete=False,environment=runtime_environment,
            limits='GPU endpoint checks do not rule out transient contention. Submodule timing and full power profiles are not measured.'))
        print('[TIMING END]',status,out,flush=True);return 0 if not failures else 2
    except BaseException as e:
        write_json(out/'failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--quality-dir',required=True);ap.add_argument('--out-dir',required=True);a=ap.parse_args();return run(a.plan,a.quality_dir,a.out_dir)
if __name__=='__main__':raise SystemExit(main())
