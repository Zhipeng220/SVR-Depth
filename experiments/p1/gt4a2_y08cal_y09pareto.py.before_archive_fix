#!/usr/bin/env python3
"""Shared Y08-Cal / Y09-Pareto development suite. No final-test claims.

Install all Python files beside this file in experiments/p1. Existing reviewed
Experiment1, GT2 and Y04-A sources remain unchanged. See README for full scope.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import contextlib
import fcntl
import gc
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import zipfile
from types import SimpleNamespace
from y08y09_common import *

ROOT=Path(__file__).resolve().parents[2]
HELPER_SHA='6fbe1a2464a7f9756f9ea68cf05f3476a89b005b8e6f889dd1a0093f231c3116'
O_SHA='558e1850d7b08939bef0db0eb4f7a48fdff8a2fb58417cd4c380ee3c6a6f6948'
MODEL_SOURCE={
 'velodepth/models/velodepth.py':'739c5f27a749174987b9a36edaac32fb98070e87dbef0a0a52f7430d61149479',
 'velodepth/models/encoder_velo.py':'3c5aa2a3a6dce03b68373953037fe8a0313a2cab36d66e5e3b88a56ec4a7ba8a'}


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT',''))
    p.add_argument('--y04-dir',default='outputs/gt4a2_y04_validation_fix3_20261025_225902')
    p.add_argument('--y04a-dir',default='auto')
    for s in SEQS:p.add_argument('--'+s+'-pair-csv',default=f'outputs/gt4a2_exp1/gt4a2_exp1_{s}_pairs_used.csv')
    p.add_argument('--out-dir',default='outputs/gt4a2_y08cal_y09pareto/validation')
    p.add_argument('--resume',action='store_true')
    g=p.add_mutually_exclusive_group(); g.add_argument('--preflight-only',action='store_true');g.add_argument('--smoke-only',action='store_true')
    p.add_argument('--stage',choices=('all','train','quality','timing','report'),default='all')
    a=p.parse_args(); require(bool(a.snapshot),'Set VDEEP_SNAPSHOT or pass --snapshot')
    require(not(a.resume and (a.smoke_only or a.preflight_only)),'Resume is for formal runs only')
    return a


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def preflight(a):
    hp=ROOT/'experiments/p1/gt4a2_y04a_native_pre_readout_eval.py'
    require(hp.is_file() and digest(hp)==HELPER_SHA,'Original reviewed Y04-A helper is missing/changed')
    helper=import_file('y08y09_original_y04a',hp)
    qa=SimpleNamespace(snapshot=a.snapshot,y04_dir=a.y04_dir,p_checkpoint='',seqs=list(SEQS),smoke_only=False,
        **{s+'_pair_csv':getattr(a,s+'_pair_csv') for s in SEQS})
    base,manifests,refs,rt,p_state=helper.preflight(qa)
    candidates=list((ROOT/'outputs').glob('gt4a2_y04a_validation*')) if a.y04a_dir=='auto' else [Path(a.y04a_dir).resolve()]
    valid=[]
    for d in candidates:
        try:
            q=json_read(d/'y04a_summary.json')
            if q.get('status')=='MECHANICS_PASS' and q.get('full_three_sequence_evaluation') and q.get('script_sha256')==HELPER_SHA and q.get('checkpoint_sha256')==helper.P_SHA:valid.append(d)
        except (OSError,ValueError):pass
    require(bool(valid),'Completed Y04-A result required; pass --y04a-dir')
    y04a=max(valid,key=lambda d:(d/'y04a_summary.json').stat().st_mtime_ns).resolve()
    prior=json_read(y04a/'y04a_preflight.json')
    require(all(prior[k]==base[k] for k in ('source_sha256','snapshot_config_sha256','reference_csv_sha256','checkpoint_sha256')),'Y04-A input identity differs')
    for r in csv_read(y04a/'y04a_all_frame_metrics.csv'):
        refs[r['sequence'],'P_native_reference',int(r['frame_index_zero_based'])]=r
    for rel,h in MODEL_SOURCE.items():require(digest(ROOT/rel)==h,f'Auto/source audit changed: {rel}')
    cp=Path(a.y04_dir).resolve()/'training/O_multi/Y04_O_multi_final_step1000.pt'
    require(cp.is_file() and digest(cp)==O_SHA,'O_multi checkpoint differs from completed Y04')
    obj=rt.torch.load(cp,map_location='cpu',weights_only=False)
    o_state=obj['adapter_state_dict']
    from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead
    for label,state in [('P',p_state),('O_multi',o_state)]:
        require(tuple(state['blocks.0.0.weight'].shape)==(32,512,1,1),f'{label} expected 512 channels')
        head=PersistentFeatureRepairHead(channels=512,bottleneck=32,levels=4)
        head.load_state_dict(state,strict=True)
        require(sum(x.numel() for x in head.parameters())==170496,'Head architecture changed')
        require(all(rt.torch.isfinite(x).all().item() for x in state.values()),'Nonfinite old head')
    source_paths=[hp,Path(__file__),Path(__file__).with_name('y08y09_common.py'),Path(__file__).with_name('y08y09_models.py'),
                  Path(__file__).with_name('test_y08y09.py')]
    if (ROOT/'run_y08cal_y09pareto.sh').exists():source_paths.append(ROOT/'run_y08cal_y09pareto.sh')
    inputs={str(p.resolve()):digest(p) for p in source_paths}
    inputs.update({str(y04a/name):digest(y04a/name) for name in ('y04a_summary.json','y04a_preflight.json','y04a_all_frame_metrics.csv')})
    inputs.update({str(ROOT/r):h for r,h in {**base['source_sha256'],**MODEL_SOURCE}.items()})
    inputs.update({base['checkpoint']:base['checkpoint_sha256'],str(cp):O_SHA,
        str(Path(a.snapshot).resolve()/'model.safetensors'):helper.MODEL_SHA,
        str(Path(a.snapshot).resolve()/'config.json'):base['snapshot_config_sha256'],
        str(Path(a.y04_dir).resolve()/'y04_all_frame_metrics.csv'):base['reference_csv_sha256']})
    for s in SEQS:
        q=Path(getattr(a,s+'_pair_csv')).resolve(); inputs[str(q)]=digest(q)
        for row in manifests[s]:
            for key in ('rgb_path','depth_path'):
                if key=='depth_path' and str(row.get('gt_match_status','')).upper()!='MATCHED':continue
                p=Path(row[key]).resolve()
                if str(p) not in inputs:inputs[str(p)]=digest(p)
    order=training_order(); counts=[]
    for s,c in SLOTS:
        counts.append(sum(str(manifests[s][i].get('gt_match_status','')).upper()=='MATCHED' for i in range(c*32+1,c*32+32)))
    require(sum(counts)==740 and sum(counts[i] for i in order)==30834,'Shared training exposure changed')
    snapshot_config=json_read(Path(a.snapshot)/'config.json')['model']
    # Exact defaults in this audited constructor: alpha_luma/beta_luma fall back
    # to FLOW_ALPHA/FLOW_BETA too. Do not silently substitute LUMA constants.
    auto={k:float(snapshot_config.get(k,v)) for k,v in
          [('alpha_flow',.1),('beta_flow',.15),('alpha_luma',.1),('beta_luma',.15)]}
    settings=plan()
    if a.smoke_only:
        settings=[config('A'),config('P'),config('O_multi'),config('B2'),config('O_fast'),
                  config('A','Auto'),config('P','Auto'),config('A','K64'),config('O_fast','K64',1.,4)]
    protocol=dict(version=VERSION,partial_smoke=a.smoke_only,input_sha256=inputs,auto_thresholds=auto,
        software=dict(torch=str(rt.torch.__version__),cuda=str(rt.torch.version.cuda),numpy=rt.np.__version__,
                      opencv=rt.cv2.__version__,python=sys.version.split()[0]),
        configuration_plan=settings,raw_configuration_count=len(settings),
        calibration_variants=['raw','cal'],formal_training_steps=1000,training_order=order,
        training_unique_GT_frames=740,supervised_exposures_per_new_head=30834,
        newly_trained_heads=['B2','O_fast'],frozen_heads=['P','O_multi'],
        train_RGB_ranges={s:[0,255] for s in SEQS},tail_start=TAIL,
        calibration='global per-config train-only OLS across three prefixes; shared 740 RGB frames, not video-specific',
        selection_budget='8 schedule/gain/frequency points x raw/cal per learned family; A has four schedules x raw/cal',
        gain_zero='physical alias of A at same schedule, never rerun or counted as a new learned candidate',
        raw_P='original P_multi, pre_readout gain1 unless configured otherwise; no Y08-H retraining',
        risk='common GT frames / raw A K32 and common 32-RGB windows; matched-schedule references additionally reported',
        dominance_axes=['macro AbsRel','macro RMSE','macro delta1 (higher is better)','worst-video AbsRel',
                        'macro harmed-frame fraction vs raw A K32','macro video-level frame-AbsRel p95','macro median-trial latency'],
        predictions='float32 NPZ for training-prefix and scored-tail frames; all RGB frames still propagate',
        timing='serial RGB-read-to-CPU-depth latency; three rotated trials; GT/metrics/writes/debug outside timer',
        splits=dict(old_three_videos='development only',independent_development=[],sealed_final_test=[],
                    independent_generalization_claim=False),
        limits=['Development screening, not Y11 or CVPR final testing',
                'Frozen K32-trained P/O are stress-tested at K16/K64/Auto, not retrained per schedule',
                'One new-head seed; matched update/GT exposure does not imply equal actual training compute',
                'O_fast is a 10465-parameter RGB/depth CNN baseline, not all possible efficient output methods',
                'Dominance is over these measured settings only; failure does not prove no possible persistent operator works'])
    return protocol,manifests,refs,rt,p_state,o_state,helper


def atomic_save(rt,obj,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    rt.torch.save(obj,q);q.replace(p)


def save_npz(np,path,**values):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    with q.open('wb') as f:np.savez_compressed(f,**values)
    q.replace(p)


def seed(rt):
    rt.torch.manual_seed(SEED);rt.torch.cuda.manual_seed_all(SEED);rt.np.random.seed(SEED);random.seed(SEED)


def make_model(a,rt,cfg,p_state,o_state,trained):
    from y08y09_models import GainHead,ChannelAffine,FastDepthAdapter
    mode='pre_readout' if cfg['family']=='P' else 'off'
    m=rt.gt2.load_model(Path(a.snapshot).resolve(),mode,rt.torch.device('cuda'))
    adapter=None
    if cfg['family']=='P':
        head=m.pixel_decoder.persistent_feature_repair;head.load_state_dict(p_state,strict=True)
        m.pixel_decoder.persistent_feature_repair=GainHead(head,cfg['gain'])
    elif cfg['family']=='O_multi':
        from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead
        adapter=PersistentFeatureRepairHead(channels=512,bottleneck=32,levels=4).cuda()
        adapter.load_state_dict(o_state,strict=True)
    elif cfg['family'] in ('B2','O_fast'):
        adapter=(ChannelAffine() if cfg['family']=='B2' else FastDepthAdapter()).cuda()
        adapter.load_state_dict(trained[cfg['family']],strict=True)
    if cfg['schedule']=='Auto':
        for k,v in a.protocol['auto_thresholds'].items():setattr(m,k,v)
    m.eval()
    for p in m.parameters():p.requires_grad_(False)
    if adapter is not None:
        adapter.eval()
        for p in adapter.parameters():p.requires_grad_(False)
    return m,adapter


def get_gt(rt,rec,shape,out,seq,index):
    require(str(rec.get('gt_match_status','')).upper()=='MATCHED','GT requested on unmatched RGB')
    path=out/'gt'/seq/f'{index:06d}.npz'
    if path.exists():
        with rt.np.load(path) as z:return z['gt']
    raw=rt.cv2.imread(str(rec['depth_path']),rt.cv2.IMREAD_UNCHANGED)
    require(raw is not None and raw.dtype==rt.np.uint16 and tuple(raw.shape)==tuple(shape['original_shape']),'GT geometry/dtype differs')
    gt=raw.astype(rt.np.float32)/5000.
    save_npz(rt.np,path,gt=gt,valid_mask=(gt>0)&rt.np.isfinite(gt),
        rgb_timestamp=rt.np.array(float(rec['rgb_timestamp'])),rgb_file=rt.np.array(Path(rec['rgb_path']).name),
        gt_file=rt.np.array(Path(rec['depth_path']).name))
    return gt


def build_training_cache(a,rt,manifests,out,p_state,o_state,signature,refs):
    from y08y09_models import Stream,rgb_for_output
    paths={}; started=time.perf_counter(); recomputed=0
    for seq in SEQS:
        all_good=True
        for c in range(8):
            path=out/'training/cache'/seq/f'cycle_{c:02d}.pt'; meta=path.with_suffix('.json')
            if not(path.exists() and meta.exists()):all_good=False;continue
            obj=json_read(meta);require(obj['signature']==signature and obj['sha256']==digest(path),'Training cache identity changed')
        if not all_good:
            m,_=make_model(a,rt,config('A'),p_state,o_state,{})
            shape=rt.exp1.shape_info_for(m,Path(manifests[seq][0]['rgb_path']),rt.torch.device('cuda'))
            stream=Stream(rt,m,config('A'),shape,debug=True); frames=[]
            with rt.torch.no_grad():
                for i,rec in enumerate(manifests[seq][:256]):
                    frame=rt.exp1.load_frame(rec,shape,rt.torch.device('cuda'),need_gt=False)
                    static=None
                    if i%32 and stream.state is not None:
                        from y08y09_models import transition
                        static=transition(rt,m,stream.previous,frame,shape,stream.dis)
                    with rt.torch.autocast('cuda',dtype=rt.torch.float16):pred,info=stream.advance(frame,i,static)
                    if is_fit_frame(i,str(rec.get('gt_match_status','')).upper()=='MATCHED'):
                        gt=get_gt(rt,rec,shape,out,seq,i)
                        mm=metrics(pred.detach().float().cpu().numpy(),gt,rt.np)
                        ref=refs[seq,'A_native',i]
                        require(max(abs(mm[k]-float(ref[k])) for k in ('AbsRel','RMSE','delta1'))<=2e-5,'Training cache native A trajectory changed')
                        frames.append(dict(index=i,features=[x.detach().cpu().clone() for x in stream.state[2]],
                            rays=stream.state[3].detach().cpu().clone(),depth=pred.detach().float().cpu().clone(),
                            rgb_small=rgb_for_output(frame,shape).detach().float().cpu().clone(),
                            gt=rt.torch.from_numpy(gt.copy()),valid_pixels=int(((gt>0)&rt.np.isfinite(gt)).sum())))
                    if i%32==31:
                        c=i//32;path=out/'training/cache'/seq/f'cycle_{c:02d}.pt'
                        if not path.exists() or not path.with_suffix('.json').exists():
                            atomic_save(rt,dict(signature=signature,sequence=seq,cycle=c,frames=frames,
                                network_shape=shape['network_shape'],padded_shape=shape['padded_shape'],paddings=shape['paddings']),path)
                            json_write(path.with_suffix('.json'),dict(signature=signature,sha256=digest(path),supervised=len(frames)))
                            recomputed+=1
                        else:
                            require(path.with_suffix('.json').exists(),'Incomplete cache file without identity; remove only that new cache file and resume')
                        print(f'[TRAIN CACHE] {seq}/{c}: {len(frames)} GT frames',flush=True);frames=[]
            del stream,m;gc.collect();rt.torch.cuda.empty_cache()
        for c in range(8):paths[seq,c]=out/'training/cache'/seq/f'cycle_{c:02d}.pt'
    event=dict(current_invocation_sec=time.perf_counter()-started,newly_completed_cycles=recomputed,
        note='Shared native-A precompute used by both new heads; include every invocation in total training cost.')
    with (out/'training/cache_compute_events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    return paths


def rng_state(rt):
    return dict(torch=rt.torch.get_rng_state(),cuda=rt.torch.cuda.get_rng_state(),np=rt.np.random.get_state(),py=random.getstate())


def restore_rng(rt,r):
    rt.torch.set_rng_state(r['torch']);rt.torch.cuda.set_rng_state(r['cuda']);rt.np.random.set_state(r['np']);random.setstate(r['py'])


def train_head(a,rt,family,paths,out,signature,p_state,o_state):
    from y08y09_models import ChannelAffine,FastDepthAdapter,readout_radius
    directory=out/'training'/family;directory.mkdir(parents=True,exist_ok=True)
    final=directory/'final.pt';recovery=directory/'recovery.pt'
    count=3 if a.smoke_only else STEPS
    if final.exists():
        obj=rt.torch.load(final,map_location='cpu',weights_only=False)
        require(obj['signature']==signature and obj['steps']==count,'Existing final head identity differs')
        return obj['state'],json_read(directory/'summary.json')
    seed(rt);m,_=make_model(a,rt,config('A'),p_state,o_state,{})
    # Freeze base readout; retain gradients from its inputs to B2 parameters.
    head=(ChannelAffine() if family=='B2' else FastDepthAdapter()).cuda();head.train()
    optimizer=rt.torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4)
    scaler=rt.torch.amp.GradScaler('cuda');order=[0,8,16] if a.smoke_only else training_order()
    curve=[];step0=0;exposure=0;resumes=[]
    initial=rt.exp1.state_sha256({k:v.detach().cpu() for k,v in head.state_dict().items()})
    if recovery.exists():
        require(a.resume,'Recovery exists; rerun with --resume')
        obj=rt.torch.load(recovery,map_location='cpu',weights_only=False)
        require(obj['signature']==signature and obj['initial_sha']==initial,'Recovery training identity differs')
        head.load_state_dict(obj['state']);optimizer.load_state_dict(obj['optimizer']);scaler.load_state_dict(obj['scaler'])
        curve=obj['curve'];step0=obj['step'];exposure=obj['exposure'];restore_rng(rt,obj['rng'])
        validate_recovery(curve,step0,order,exposure)
        resumes=obj['resumes']+[dict(restored_step=step0,time_unix=time.time())]
        print(f'[RESUME {family}] committed step {step0}',flush=True)
    journal=directory/'physical_updates.jsonl';started=time.perf_counter();rt.torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(step0+1,count+1):
            slot=order[step-1];seq,c=SLOTS[slot];cp=paths[seq,c]
            require(digest(cp)==json_read(cp.with_suffix('.json'))['sha256'],'Training cache changed')
            cycle=rt.torch.load(cp,map_location='cpu',weights_only=False)
            m.pixel_decoder.radial_module.set_original_shapes(tuple(cycle['network_shape']))
            optimizer.zero_grad(set_to_none=True);losses=[];n=len(cycle['frames']);start=time.perf_counter()
            for begin in range(0,n,4):
                terms=[]
                for item in cycle['frames'][begin:begin+4]:
                    require(is_fit_frame(item['index']),'Training leakage outside shared prefixes')
                    gt=item['gt'].cuda();native=item['depth'].cuda()
                    with rt.torch.autocast('cuda',dtype=rt.torch.float16):
                        if family=='O_fast':pred=head(native,item['rgb_small'].cuda())
                        else:
                            features=[x.cuda() for x in item['features']]
                            radius=readout_radius(m,head(features))
                            pred=rt.gt2.make_z_depth(radius,item['rays'].cuda(),cycle['padded_shape'],cycle['paddings'],m.interpolation_mode)
                        valid=(gt>0)&rt.torch.isfinite(gt)
                        require(bool(rt.torch.all(rt.torch.isfinite(pred[valid])&(pred[valid]>0)).item()),'Invalid training prediction')
                        loss,npix=rt.gt2.masked_log_depth_loss(pred,gt)
                        if step==1:
                            require(float((pred.detach().float()-native.float()).abs().max().item())<=1e-3,'New head initialization must reproduce native depth')
                        require(npix==item['valid_pixels'] and rt.torch.isfinite(loss).item(),'Changed training mask / nonfinite loss')
                        terms.append(loss);losses.append(float(loss.detach().item()))
                    exposure+=1
                scaler.scale(rt.torch.stack(terms).sum()/n).backward()
            scaler.unscale_(optimizer)
            gradients=[p.grad for p in head.parameters() if p.grad is not None]
            require(gradients and all(rt.torch.isfinite(g).all().item() for g in gradients),'Nonfinite/missing gradients')
            norm=float(rt.torch.nn.utils.clip_grad_norm_(head.parameters(),1.).item())
            old=float(scaler.get_scale());scaler.step(optimizer);scaler.update()
            require(float(scaler.get_scale())>=old,'AMP skipped optimizer update')
            row=dict(step=step,slot=slot,sequence=seq,cycle=c,supervised_frames=n,loss=statistics.mean(losses),gradient_norm=norm,wall_sec=time.perf_counter()-start)
            curve.append(row)
            with journal.open('a') as f:f.write(json.dumps(row)+'\n')
            if step==1 or step%10==0 or step==count:print(f'[TRAIN {family}] {step}/{count} loss={row["loss"]:.6f}',flush=True)
            if step%25==0 or step==count:
                atomic_save(rt,dict(signature=signature,initial_sha=initial,step=step,exposure=exposure,
                    state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(),curve=curve,rng=rng_state(rt),resumes=resumes),recovery)
                csv_write(directory/'curve.csv',curve)
        require(a.smoke_only or exposure==30834,'Training exposure no longer matches Y04 multi')
        state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        final_sha=rt.exp1.state_sha256(state);require(final_sha!=initial,'Head never changed')
        physical=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        summary=dict(signature=signature,steps=count,supervised_exposures=exposure,initial_sha=initial,final_sha=final_sha,
            params=sum(p.numel() for p in head.parameters()),training_order_sha=digest_order(order),
            physical_completed_updates=len(physical),physical_supervised_exposures=sum(r['supervised_frames'] for r in physical),
            sum_logged_update_wall_sec=sum(r['wall_sec'] for r in physical),current_invocation_training_sec=time.perf_counter()-started,
            peak_cuda_allocated_gib=rt.torch.cuda.max_memory_allocated()/1024**3,resumes=resumes,
            training_state='frozen native A K32; independently trained output-only; no writeback',
            note='Updates/GT exposure match P_multi; actual compute differs. Shared cache build cost reported separately.')
        # Summary first: a process crash can safely repeat finalization from recovery.
        json_write(directory/'summary.json',summary)
        atomic_save(rt,dict(signature=signature,steps=count,state=state,summary=summary),final)
        return state,summary
    finally:
        del head,m,optimizer,scaler;gc.collect();rt.torch.cuda.empty_cache()


def digest_order(order):
    import hashlib
    return hashlib.sha256(','.join(map(str,order)).encode()).hexdigest()


def audit_auto(a,rt,manifests,out,p_state,o_state,signature):
    """Real upstream encode_decode_test equivalence, including forced restarts."""
    from y08y09_models import Stream,transition
    done=out/'auto_equivalence.json'
    if done.exists():
        old=json_read(done);require(old['signature']==signature and old['passed'],'Auto audit identity differs');return
    reports=[]
    for seq,forced in [(s,False) for s in SEQS]+[('crowd2',True)]:
        cfg=config('A','Auto');m,_=make_model(a,rt,cfg,p_state,o_state,{})
        original={k:getattr(m,k) for k in a.protocol['auto_thresholds']}
        if forced:m.alpha_flow=-1.;m.beta_flow=0.
        n=8 if forced else 16
        shape=rt.exp1.shape_info_for(m,Path(manifests[seq][0]['rgb_path']),rt.torch.device('cuda'))
        frames=[rt.exp1.load_frame(rec,shape,rt.torch.device('cuda'),need_gt=False) for rec in manifests[seq][:n]]
        stream=Stream(rt,m,cfg,shape,debug=True);statics=[]
        with rt.torch.no_grad(),rt.torch.autocast('cuda',dtype=rt.torch.float16):
            for i in range(1,n):statics.append(transition(rt,m,frames[i-1],frames[i],shape,stream.dis))
            image=rt.torch.cat([f['img'] for f in frames]);flow=rt.torch.cat([s['raw_flow_px'] for s in statics]+[rt.torch.zeros_like(statics[-1]['raw_flow_px'])])
            packed=dict(image=image,image_original=image.clone(),flow_bwd=flow,
                validity_mask=shape['validity_mask'].repeat(n,1,1,1),depth_paddings=shape['depth_paddings'].repeat(n,1))
            _,reference=m.encode_decode_test(packed,[dict(B=1,T=n)])
            refs={k:reference[k].detach().float().cpu() for k in ('radius','confidence','rays')}
            del reference,packed,image,flow
            differences=[]
            for i,frame in enumerate(frames):
                stream.advance(frame,i,None if i==0 else statics[i-1])
                r,c,f,rays=stream.state
                differences.append(max(float((x.detach().float().cpu()-refs[k][i:i+1]).abs().max().item())
                    for k,x in [('radius',r),('confidence',c),('rays',rays)]))
            require(max(differences)<=1e-3,f'Original Auto equivalence failed: {seq}, forced={forced}, max={max(differences)}')
            if forced:require(stream.counts['rejected_candidates']==n-1,'Forced Auto audit did not cover rejection/refresh')
        reports.append(dict(sequence=seq,audit_only_forced_restart=forced,n_RGB=n,
            max_abs_radius_confidence_rays=max(differences),counts=stream.counts))
        for k,v in original.items():setattr(m,k,v)
        del stream,m,frames,statics,refs;gc.collect();rt.torch.cuda.empty_cache()
    json_write(done,dict(signature=signature,passed=True,reports=reports,
        note='Forced thresholds only test rejection/overlap semantics; never used in quality or timing.'))


def raw_path(out,cfg,seq,index):
    return out/'predictions'/cfg['id']/seq/f'{index:06d}.npz'


def quality_unit(a,rt,cfg,seq,manifest,out,trained,p_state,o_state,refs,signature):
    from y08y09_models import Stream
    directory=out/'quality'/cfg['id'];directory.mkdir(parents=True,exist_ok=True)
    meta_path=directory/(seq+'.json');rows_path=directory/(seq+'.csv')
    if meta_path.exists():
        meta=json_read(meta_path)
        require(meta['signature']==signature and meta['rows_sha256']==digest(rows_path),'Quality unit identity differs')
        for p,h in meta['prediction_sha256'].items():require(Path(p).exists() and digest(p)==h,'Saved prediction missing/changed')
        return meta
    m,adapter=make_model(a,rt,cfg,p_state,o_state,trained)
    shape=rt.exp1.shape_info_for(m,Path(manifest[0]['rgb_path']),rt.torch.device('cuda'))
    stream=Stream(rt,m,cfg,shape,adapter,debug=True)
    native_reference={}
    if cfg['family'] in ('B2','O_multi','O_fast'):
        base_path=out/'quality'/config('A',cfg['schedule'])['id']/(seq+'.csv')
        native_reference={r['index']:r for r in typed_rows(base_path)}
    rows=[];stats=Moments();files={};n=min(16,len(manifest)) if a.smoke_only else len(manifest)
    t=time.perf_counter()
    with rt.torch.no_grad():
        for i,rec in enumerate(manifest[:n]):
            pred,meta=stream.step(i,rec);matched=str(rec.get('gt_match_status','')).upper()=='MATCHED'
            require(tuple(pred.shape)==tuple(shape['original_shape']),'Prediction geometry differs')
            record=dict(config_id=cfg['id'],variant=cfg['id']+'/raw',sequence=seq,index=i,
                matched=matched,scored=a.smoke_only or i>=TAIL[seq],is_fit_frame=is_fit_frame(i,matched),
                rgb_file=Path(rec['rgb_path']).name,gt_file=Path(rec['depth_path']).name if matched else '',
                rgb_timestamp=float(rec['rgb_timestamp']),**meta)
            if matched and keep_prediction(seq,i):
                gt=get_gt(rt,rec,shape,out,seq,i);record.update(metrics(pred,gt,rt.np))
                if is_fit_frame(i):stats.add(pred,gt,rt.np)
                # Historical guards only for exact original fixed-K settings.
                arm={'A':'A_native','O_multi':'Y04_O_multi','P':'P_native_reference'}.get(cfg['family'])
                if cfg['schedule']=='K32' and cfg['frequency']==1 and (cfg['family']=='A' or cfg['gain']==1.) and arm:
                    ref=refs[seq,arm,i]
                    require(int(ref['valid_pixels'])==record['valid_pixels'],'Historical mask pixel count differs')
                    require(max(abs(record[k]-float(ref[k])) for k in ('AbsRel','RMSE','delta1'))<=2e-5,
                        f'Original A/O trajectory changed: {cfg["id"]}/{seq}/{i}')
                if cfg['family'] in ('B2','O_multi','O_fast'):
                    native=stream.native_prediction.detach().float().cpu().numpy()
                    mm=metrics(native,gt,rt.np);ref=native_reference[i]
                    require(max(abs(mm[k]-float(ref[k])) for k in ('AbsRel','RMSE','delta1'))<=2e-5,'Output-only native A trajectory changed')
            if native_reference:
                ref=native_reference[i]
                require(meta['keyframe']==ref['keyframe'] and meta['age']==ref['age'],'Output-only Auto/K scheduler changed')
            if keep_prediction(seq,i):
                p=raw_path(out,cfg,seq,i)
                save_npz(rt.np,p,depth=pred,index=rt.np.array(i),rgb_timestamp=rt.np.array(record['rgb_timestamp']),
                    keyframe=rt.np.array(meta['keyframe']),age=rt.np.array(meta['age']))
                files[str(p)]=digest(p)
            rows.append(record)
            if i==0 or (i+1)%100==0 or i+1==n:print(f'[QUALITY] {cfg["id"]} {seq} {i+1}/{n}',flush=True)
    csv_write(rows_path,rows)
    complete=dict(signature=signature,rows_sha256=digest(rows_path),prediction_sha256=files,
        counts=stream.counts,processed_RGB=n,network_shape=list(shape['network_shape']),original_shape=list(shape['original_shape']),
        fit_moments=stats.__dict__,diagnostic_wall_sec=time.perf_counter()-t,
        peak_memory_is_not_latency=True)
    json_write(meta_path,complete)
    del stream,m,adapter;gc.collect();rt.torch.cuda.empty_cache()
    return complete


def typed_rows(path):
    rows=csv_read(path)
    for r in rows:
        for k in ('index','age','last_keyframe'):r[k]=int(r[k])
        for k in ('matched','scored','is_fit_frame','keyframe','output_adapter_applied','repair_requested'):
            r[k]=str(r[k]).lower()=='true'
        for k in ('AbsRel','RMSE','delta1'):
            if r.get(k) not in ('',None):r[k]=float(r[k])
        if r.get('valid_pixels') not in ('',None):r['valid_pixels']=int(r['valid_pixels'])
    return rows


def fit_and_calibrate(a,rt,settings,out,signature):
    calibration={};all_rows=[]
    for cfg in settings:
        acc=Moments();base_rows=[]
        for seq in SEQS:
            meta=json_read(out/'quality'/cfg['id']/(seq+'.json'))
            m=Moments();m.__dict__.update(meta['fit_moments']);acc.merge(m)
            base_rows.extend(typed_rows(out/'quality'/cfg['id']/(seq+'.csv')))
        calibration[cfg['id']]=acc.finish();json_write(out/'calibration'/f'{cfg["id"]}.json',dict(signature=signature,**acc.finish()))
        for r in base_rows:
            all_rows.append(r)
            c=dict(r,variant=cfg['id']+'/cal')
            if r['matched'] and keep_prediction(r['sequence'],r['index']):
                with rt.np.load(raw_path(out,cfg,r['sequence'],r['index'])) as z:pred=z['depth']
                cp=calibration[cfg['id']]
                adjusted=rt.np.maximum(pred*cp['scale']+cp['bias_m'],FLOOR).astype(rt.np.float32)
                with rt.np.load(out/'gt'/r['sequence']/f'{r["index"]:06d}.npz') as z:gt=z['gt']
                c.update(metrics(adjusted,gt,rt.np));c['floored_pixels']=int((pred*cp['scale']+cp['bias_m']<FLOOR).sum())
            all_rows.append(c)
    csv_write(out/'y08y09_frame_metrics.csv',all_rows)
    json_write(out/'calibration.json',calibration)
    return all_rows,calibration


def gpu_endpoint(rt):
    import csv
    binary=shutil.which('nvidia-smi')
    if not binary and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():binary='/usr/lib/wsl/lib/nvidia-smi'
    result=dict(clean=False,available=False,foreign=[],unmapped=[])
    if not binary:return dict(result,error='nvidia-smi not found; quality may run, verified timing cannot be claimed')
    def query(flag,fields):
        p=subprocess.run([binary,flag+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
        return list(csv.reader(io.StringIO(p.stdout)))
    try:
        inventory=query('--query-gpu=','index,uuid,name,temperature.gpu,clocks.sm,power.draw')
        uuid=str(getattr(rt.torch.cuda.get_device_properties(0),'uuid',''))
        chosen=next((r for r in inventory if r[1].strip()==uuid),None)
        if chosen is None and len(inventory)==1 and rt.torch.cuda.device_count()==1:chosen=inventory[0]
        require(chosen is not None,'Cannot map CUDA0 to GPU monitor')
        result.update(available=True,gpu_uuid=chosen[1].strip(),name=chosen[2].strip(),temperature=chosen[3].strip(),
            sm_clock=chosen[4].strip(),power_w=chosen[5].strip())
        for r in query('--query-compute-apps=','gpu_uuid,pid,process_name'):
            if len(r)<3 or r[0].strip()!=result['gpu_uuid']:continue
            try:pid=int(r[1].strip())
            except ValueError:result['unmapped'].append(r);continue
            if pid==os.getpid():continue
            path=Path(f'/proc/{pid}/cmdline')
            name=Path(path.read_bytes().split(b'\0')[0].decode(errors='replace')).name if path.exists() else ''
            reported=Path(r[2].strip()).name
            if name and (name==reported or (name.startswith('python') and reported.startswith('python'))):result['foreign'].append(r)
            else:result['unmapped'].append(r)
        result['clean']=not result['foreign'] and not result['unmapped']
    except (OSError,ValueError,IndexError,subprocess.SubprocessError,RuntimeError) as e:result['error']=str(e)
    return result


def timing_trial(a,rt,cfg,variant,seq,manifest,repeat,position,cal,out,trained,p_state,o_state,signature):
    from y08y09_models import Stream
    mode=variant.rsplit('/',1)[1]
    directory=out/'timing'/cfg['id']/mode;directory.mkdir(parents=True,exist_ok=True)
    meta_path=directory/f'{seq}_r{repeat}.json';csv_path=meta_path.with_suffix('.csv')
    if meta_path.exists():
        old=json_read(meta_path)
        require(old['signature']==signature and old['rows_sha256']==digest(csv_path),'Timing unit identity differs')
        if old['endpoint_clean'] and old['quality_checked']:return old
        # An unverified trial is retained under a separate name, never accepted.
        meta_path.rename(meta_path.with_name(meta_path.stem+f'_unverified_{time.time_ns()}.json'))
    m,adapter=make_model(a,rt,cfg,p_state,o_state,trained)
    shape=rt.exp1.shape_info_for(m,Path(manifest[0]['rgb_path']),rt.torch.device('cuda'))
    stream=Stream(rt,m,cfg,shape,adapter,debug=False)
    n=min(16,len(manifest)) if a.smoke_only else len(manifest)
    coef=cal[cfg['id']] if mode=='cal' else None
    frames=[];checks=0;bad='';rt.torch.cuda.reset_peak_memory_stats()
    qrows={r['index']:r for r in typed_rows(out/'quality'/cfg['id']/(seq+'.csv'))}
    # Equal warmup: 64 real RGB frames, then reset complete recurrent state.
    with rt.torch.no_grad():
        # Prime identical file bytes for every trial; decode/preprocess remain timed.
        for rec in manifest[:n]:Path(rec['rgb_path']).read_bytes()
        for i,rec in enumerate(manifest[:min(16 if a.smoke_only else 64,len(manifest))]):stream.step(i,rec,coef)
        rt.torch.cuda.synchronize();stream.reset()
        before=gpu_endpoint(rt)
        require(before['clean'],f'Timing endpoint not verified clean: {before}')
        for i,rec in enumerate(manifest[:n]):
            rt.torch.cuda.synchronize();start=time.perf_counter()
            pred,meta=stream.step(i,rec,coef)
            rt.torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000.
            frames.append(dict(variant=variant,sequence=seq,index=i,repeat=repeat,position=position,
                latency_ms=elapsed,scored=a.smoke_only or i>=TAIL[seq],keyframe=meta['keyframe'],age=meta['age']))
            require(meta['keyframe']==qrows[i]['keyframe'] and meta['age']==qrows[i]['age'],'Timing scheduler diverged from quality')
            # Outside the measured interval. Every tail frame's scheduler metadata
            # is checked; sparse *depth values* are checked against full quality.
            if keep_prediction(seq,i) and (i==0 or i==n-1 or i%32 in (1,31)):
                with rt.np.load(raw_path(out,cfg,seq,i)) as z:
                    expected=z['depth'];expected_age=int(z['age']);expected_key=bool(z['keyframe'])
                if coef:expected=rt.np.maximum(expected*coef['scale']+coef['bias_m'],FLOOR).astype(rt.np.float32)
                require(expected_age==meta['age'] and expected_key==meta['keyframe'],'Timing scheduler diverged from quality')
                require(bool(rt.np.allclose(pred,expected,rtol=1e-5,atol=1e-4)),'Timing prediction diverged from quality')
                checks+=1
            if i==0 or (i+1)%200==0 or i+1==n:print(f'[TIME] r{repeat} {variant} {seq} {i+1}/{n}',flush=True)
        after=gpu_endpoint(rt)
    valid=before['clean'] and after['clean']
    require(stream.counts==json_read(out/'quality'/cfg['id']/(seq+'.json'))['counts'],'Timing call counts differ from quality')
    tail=[r['latency_ms'] for r in frames if r['scored']]
    require(bool(tail),'No common scored RGB latency frames')
    csv_write(csv_path,frames)
    summary=dict(signature=signature,variant=variant,sequence=seq,repeat=repeat,position=position,
        n_RGB=n,n_tail_RGB=len(tail),tail_mean_ms=statistics.mean(tail),tail_p50_ms=quantile(tail,.5),
        tail_p95_ms=quantile(tail,.95),whole_mean_ms=statistics.mean(r['latency_ms'] for r in frames),
        counts=stream.counts,serial_fps=1000/statistics.mean(tail),
        input_cache='RGB file bytes primed before each trial; decode and all preprocessing still timed',
        peak_allocated_gib=rt.torch.cuda.max_memory_allocated()/1024**3,
        endpoint_clean=valid,endpoints=[before,after],quality_checked=checks>0,depth_checks=checks,
        rows_sha256=digest(csv_path),timing_scope='serial read/decode/preprocess/flow/state/head/readout/depth/CPU copy/sync; no GT/metric/log/diagnostic work')
    json_write(meta_path,summary)
    del stream,m,adapter;gc.collect();rt.torch.cuda.empty_cache()
    require(valid,'Timing contaminated at endpoint; this trial is not accepted. Resume after clearing contention.')
    return summary


def run_timing(a,rt,settings,manifests,cal,out,trained,p_state,o_state,signature):
    variants=[(c,c['id']+'/'+mode) for c in settings for mode in ('raw','cal')]
    result=[];repeats=1 if a.smoke_only else 3
    for repeat in range(repeats):
        for si,seq in enumerate(SEQS):
            offset=(repeat*max(1,len(variants)//3)+si*max(1,len(variants)//len(SEQS)))%len(variants)
            ordered=variants[offset:]+variants[:offset]
            for position,(cfg,v) in enumerate(ordered):
                result.append(timing_trial(a,rt,cfg,v,seq,manifests[seq],repeat+1,position,cal,out,trained,p_state,o_state,signature))
    json_write(out/'y09_timing_trials.json',result)
    return result


def additional_risk_columns(summary,rows):
    lookup={(r['variant'],r['sequence'],r['index']):r for r in rows if r['matched'] and r['scored']}
    for s in summary:
        id_,mode=s['variant'].split('/');schedule=id_.split('_')[-3]
        reference=f'A_{schedule}_g0_f1/{mode}'
        rr=[r for r in rows if r['variant']==s['variant'] and r['sequence']==s['sequence'] and r['scored'] and r['matched']]
        paired=[(r,lookup.get((reference,s['sequence'],r['index']))) for r in rr]
        require(all(b is not None for _,b in paired),'Missing matched-schedule A reference')
        s['matched_schedule_A_reference']=reference
        s['harmed_frames_vs_matched_schedule_and_calibration_A']=sum(r['AbsRel']>b['AbsRel']+HARM_TOL for r,b in paired)


def make_review(out,include_summary=True):
    path=out/'y08cal_y09pareto_review.zip';tmp=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for pattern in ('*.json','*.csv','*.md','*.jsonl'):
            for p in sorted(out.glob(pattern)):
                if p.name!='failure.json':z.write(p,p.name)
        for prefix in ('quality','calibration','timing','training'):
            for p in sorted((out/prefix).rglob('*')):
                if p.is_file() and p.suffix in ('.json','.csv','.jsonl'):
                    z.write(p,str(p.relative_to(out)))
        for name in ('gt4a2_y08cal_y09pareto.py','y08y09_common.py','y08y09_models.py','test_y08y09.py'):
            p=Path(__file__).with_name(name)
            if p.exists():z.write(p,'source/'+name)
        protocol=json_read(out/'protocol.json')
        for path in protocol['input_sha256']:
            p=Path(path)
            if p.suffix=='.py' and (p.is_relative_to(ROOT/'experiments/p1') or p.is_relative_to(ROOT/'velodepth/models')):
                z.write(p,'source/repository/'+str(p.relative_to(ROOT)))
        if (ROOT/'run_y08cal_y09pareto.sh').exists():z.write(ROOT/'run_y08cal_y09pareto.sh','source/run_y08cal_y09pareto.sh')
    with zipfile.ZipFile(tmp) as z:require(z.testzip() is None,'Review ZIP integrity failed')
    tmp.replace(path);return path


def report(a,rt,settings,out,rows,timing,training,signature):
    quality=quality_summary(rows);additional_risk_columns(quality,rows)
    csv_write(out/'y09_sequence_quality.csv',quality)
    cal_ids={c['id'] for c in settings if c['schedule']=='K32' and c['frequency']==1}
    cal_quality=[r for r in quality if r['variant'].split('/')[0] in cal_ids]
    csv_write(out/'y08_cal_sequence_quality.csv',cal_quality)
    points=pareto_rows(quality,timing)
    require(len(points)==len(settings)*2,'Not every variant has three-sequence verified quality/timing')
    csv_write(out/'y09_pareto_points.csv',points)
    sequences={c['id']:{s:json_read(out/'quality'/c['id']/(s+'.json'))['counts'] for s in SEQS} for c in settings}
    summary=dict(version=VERSION,signature=signature,status='SMOKE_PASS' if a.smoke_only else 'DEVELOPMENT_SUITE_COMPLETE',
        partial_smoke=a.smoke_only,raw_configurations=len(settings),calibration_variants=2,quality_variants=len(points),
        quality_RGB_records=len(rows)//2,GT_tail_records=sum(r['matched'] and r['scored'] for r in rows),
        timing_trials=len(timing),expected_timing_trials=len(settings)*2*3*(1 if a.smoke_only else 3),
        all_timing_endpoints_verified=all(r['endpoint_clean'] for r in timing),
        newly_trained_heads={f:{k:v for k,v in t.items() if k in ('steps','supervised_exposures','params','final_sha','physical_completed_updates')} for f,t in training.items()},
        auto_equivalence_passed=json_read(out/'auto_equivalence.json')['passed'],quality_and_timing_checked=True,
        full_three_sequence_evaluation=not a.smoke_only,independent_testing_performed=False,
        counts=sequences,limits=a.protocol['limits'],no_automatic_idea_success=True)
    if not a.smoke_only:
        require(summary['quality_RGB_records']==36*2218 and summary['timing_trials']==648,'Formal plan incomplete')
        require(summary['GT_tail_records']==936*72,'Formal shared GT frame count changed')
        require(len({(r['variant'],r['sequence'],r['repeat']) for r in timing})==648,'Duplicate/missing timing trials')
        require(all(t['steps']==1000 and t['supervised_exposures']==30834 for t in training.values()),'Formal training budget incomplete')
    json_write(out/'summary.json',summary)
    note=('Y08-Cal and Y09-Pareto completed on the three existing DEVELOPMENT videos.\n'
          'Y08-H is not retrained. No Y09-D branch is included.\n'
          'See y08_cal_sequence_quality.csv and y09_pareto_points.csv.\n'
          'Raw float32 depth, shared GT/masks, cached states and new weights remain on the server.\n'
          'Review ZIP excludes prediction arrays, GT arrays, .pt weights/caches and optimizer state.\n'
          'Do not call these independent-scene, multi-seed, or Y11/CVPR final results.\n')
    (out/'RESULT_README.md').write_text(note)
    return make_review(out)


def main():
    a=args_parser();out=Path(a.out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    require(out!=Path(a.y04_dir).resolve() and Path(a.y04_dir).resolve() not in out.parents,'Output must be outside old Y04')
    with (out/'.run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This output directory is already running')
        had_summary=(out/'summary.json').exists()
        if had_summary and (out/'y08cal_y09pareto_review.zip').exists():
            with zipfile.ZipFile(out/'y08cal_y09pareto_review.zip') as z:
                if z.testzip() is None and json.loads(z.read('summary.json'))==json_read(out/'summary.json'):
                    print(f'[ALREADY COMPLETE] {out / "y08cal_y09pareto_review.zip"}',flush=True);return 0
        started=time.perf_counter()
        try:
            protocol,manifests,refs,rt,p_state,o_state,helper=preflight(a);a.protocol=protocol
            identity=out/'protocol.json'
            if identity.exists():require(json_read(identity)==protocol,'Input/plan identity differs; use the original files or a new output directory')
            else:
                require(not a.resume,'Cannot resume without protocol.json');json_write(identity,protocol)
            signature=digest(identity)
            print(f'[PREFLIGHT PASS] {len(protocol["configuration_plan"])} raw settings; train-only calibration; two new heads',flush=True)
            if a.preflight_only:return 0
            require(rt.torch.cuda.is_available(),'CUDA required for real experiment')
            rt.torch.set_num_threads(4);seed(rt)
            props=rt.torch.cuda.get_device_properties(0)
            env=dict(gpu_name=props.name,total_memory=props.total_memory,uuid=str(getattr(props,'uuid','')),
                capability=list(rt.torch.cuda.get_device_capability(0)),threads=rt.torch.get_num_threads(),
                cudnn_benchmark=rt.torch.backends.cudnn.benchmark,matmul_allow_tf32=rt.torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=rt.torch.backends.cudnn.allow_tf32,software=protocol['software'])
            env_path=out/'environment.json'
            if env_path.exists():require(json_read(env_path)==env,'GPU/backend environment changed; do not mix timing runs')
            else:json_write(env_path,env)
            audit_auto(a,rt,manifests,out,p_state,o_state,signature)
            trained={};training={}
            if a.stage in ('all','train'):
                paths=build_training_cache(a,rt,manifests,out,p_state,o_state,signature,refs)
                for family in ('B2','O_fast'):
                    trained[family],training[family]=train_head(a,rt,family,paths,out,signature,p_state,o_state)
            else:
                for family in ('B2','O_fast'):
                    obj=rt.torch.load(out/'training'/family/'final.pt',map_location='cpu',weights_only=False)
                    require(obj['signature']==signature,'Trained head identity differs')
                    trained[family]=obj['state'];training[family]=obj['summary']
            if a.stage=='train':print('[TRAINING COMPLETE] continue with --resume --stage all',flush=True);return 0
            settings=protocol['configuration_plan']
            if a.stage in ('all','quality'):
                for cfg in settings:
                    for seq in SEQS:quality_unit(a,rt,cfg,seq,manifests[seq],out,trained,p_state,o_state,refs,signature)
                rows,cal=fit_and_calibrate(a,rt,settings,out,signature)
            else:
                # Read tables with the same parser; full raw rows are reconstructed
                # from authoritative per-unit CSVs; calibrated metrics are recomputed.
                rows,cal=fit_and_calibrate(a,rt,settings,out,signature)
            if a.stage=='quality':
                qq=quality_summary(rows);additional_risk_columns(qq,rows);csv_write(out/'y09_sequence_quality.csv',qq)
                print('[QUALITY COMPLETE] continue with --resume --stage all',flush=True);return 0
            if a.stage in ('all','timing'):timing=run_timing(a,rt,settings,manifests,cal,out,trained,p_state,o_state,signature)
            else:timing=json_read(out/'y09_timing_trials.json')
            for p,h in protocol['input_sha256'].items():require(digest(p)==h,f'Input changed during execution: {p}')
            with (out/'invocations.jsonl').open('a') as f:f.write(json.dumps(dict(stage=a.stage,wall_sec=time.perf_counter()-started,resume=a.resume,argv=sys.argv))+'\n')
            result=report(a,rt,settings,out,rows,timing,training,signature)
            (out/'failure.json').unlink(missing_ok=True)
            print(f'[{"SMOKE_PASS" if a.smoke_only else "DEVELOPMENT_SUITE_COMPLETE"}] {result}',flush=True)
            return 0
        except BaseException as e:
            # A completed summary must never survive a failed archive/report phase.
            if not had_summary:(out/'summary.json').unlink(missing_ok=True)
            json_write(out/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),resume_command=f'bash run_y08cal_y09pareto.sh --resume {out}'))
            raise


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:print('Interrupted; resume from printed output directory.',file=sys.stderr);raise SystemExit(130)
    except Exception:traceback.print_exc();raise SystemExit(1)
