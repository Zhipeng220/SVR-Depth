#!/usr/bin/env python3
"""Y08-H: one fresh P_multi training run, adding a train-only A-relative hinge.

Same initial head, 24 cycle slots, 1000 committed updates, four-frame TBPTT.
Native pre_readout/gain1/K32 inference. No tail-dependent checkpoint selection.
Requires the unchanged Y04-A, Y07-W, Y04-Fix3 and C3 scripts already installed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = "Y08-H-1.0"
ARM = "Y08H_P_multi_health_lambda1"
Y07_SHA = "e3ef2dae3bc387ddb9f9e11828d266bb14aaab9be1b17b9390666a8ac38ad358"
C3_SHA = "76327208acac4543b2a87ccba8b7af435e4c5ce65267b42052e51b435cbad7f5"
Y04_SHA = "2a24e5a67487d55177e3374f9c6be5e92795b508dfcdb299cefe46e37cc31d4f"
INIT_SHA = "3efaa4354f1a3930d771d09ef334baccfa69fe048e741040e7769c2ec1459e0f"
ORDER_SHA = "aa7aff6213fdbf9ddce8703e2eb707e8cb8eab495a575f96698d9d25d9b9d934"
STEPS, K, TBPTT, SEED, LAMBDA = 1000, 32, 4, 20261025, 1.0
SEQUENCES = ("crowd2", "crowd3", "balloon2")
SLOTS = [(s,c) for s in SEQUENCES for c in range(8)]


def require(test, message):
    if not test: raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_module(name, rel, expected):
    path=ROOT/rel
    require(path.is_file() and sha(path)==expected, f"Required source missing/changed: {rel}")
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT',''))
    p.add_argument('--y04-dir',default='outputs/gt4a2_y04_validation_fix3_20261025_225902')
    p.add_argument('--y04a-dir',default='auto')
    p.add_argument('--initial-head',default='outputs/gt4a2_c3/gt4a2_c3_initial_head.pt')
    for seq in SEQUENCES:p.add_argument(f'--{seq}-pair-csv',default=f'outputs/gt4a2_exp1/gt4a2_exp1_{seq}_pairs_used.csv')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--preflight-only',action='store_true')
    modes.add_argument('--smoke-only',action='store_true')
    modes.add_argument('--self-test',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--out-dir',default='outputs/gt4a2_y08h_validation')
    a=p.parse_args(argv);a.seqs=list(SEQUENCES)
    if not a.self_test and not a.snapshot:p.error('Set VDEEP_SNAPSHOT or pass --snapshot')
    if a.resume and (a.smoke_only or a.preflight_only):p.error('--resume is for a formal incomplete run only')
    return a


def sample_order():
    rng=random.Random(SEED);order=[]
    while len(order)<STEPS:
        block=list(range(24));rng.shuffle(block);order.extend(block)
    return order[:STEPS]


def training_reference(refs, seq, idx):
    require(seq in SEQUENCES and 0<=idx<256 and idx%K!=0, f'Training target outside locked propagation prefix: {seq}/{idx}')
    return refs[seq,idx]


def health_hinge(torch, pred, gt, baseline):
    """Differentiate current AbsRel; baseline is a detached historical training scalar."""
    p,g=pred.float(),gt.float()
    valid=(g>0)&torch.isfinite(g)
    require(int(valid.sum().item())>0,'No GT-valid pixels for health loss')
    require(bool(torch.all(torch.isfinite(p[valid]) & (p[valid]>0)).item()),'Invalid prediction; refusing to shrink GT mask')
    absrel=((p[valid]-g[valid]).abs()/g[valid]).mean()
    return absrel,(absrel-float(baseline)).relu()


def cache_signature(helper, base, seq):
    return {'format':'c3-f32-context-v1','pretrained_weights_sha256':helper.MODEL_SHA,
        'pair_sha256':base['manifests'][seq]['sha256'],
        'original_source_hashes':{rel:base['input_sha256'][str(ROOT/rel)] for rel in helper.SOURCE_SHA}}


def preflight(a, y7, helper):
    base,manifests,refs,rt,_trained_P=y7.preflight(a,helper)
    del _trained_P  # Never initialize H from the final P weights.
    c3=load_module('y08h_c3','experiments/p1/gt4a2_c3_closed_loop.py',C3_SHA)
    y4=load_module('y08h_y04','experiments/p1/gt4a2_y04_fair_multiscene.py',Y04_SHA)
    require((c3.REFRESH,c3.TBPTT,c3.LR)==(K,TBPTT,1e-4),'C3 protocol constants differ')
    initpath=Path(a.initial_head).resolve()
    require(initpath.is_file(),f'Missing original common initial head: {initpath}; do not substitute final weights')
    initial=rt.torch.load(initpath,map_location='cpu',weights_only=False)
    require(rt.exp1.state_sha256(initial)==INIT_SHA,'Common initial head tensor SHA differs')
    head=y7.build_head(initial);del head
    for key,value in initial.items():
        if key.endswith('.6.weight') or key.endswith('.6.bias'):
            require(int(rt.torch.count_nonzero(value).item())==0,'Expected zero-residual common initialization')
    order=sample_order()
    require(hashlib.sha256(','.join(map(str,order)).encode()).hexdigest()==ORDER_SHA,'Sample-order reconstruction differs')
    ps=helper.json_read(Path(a.y04_dir)/'training/P_multi/gt4a2_c3_C_training_summary.json')
    require(ps['initial_head_state_sha256']==INIT_SHA and ps['sample_order_sha256']==ORDER_SHA,'Original P training budget binding differs')
    curvepath=Path(a.y04_dir)/'training/P_multi/gt4a2_c3_C_training_curve.csv'
    oldcurve=helper.csv_read(curvepath)
    require({int(r['step']) for r in oldcurve}==set(range(1,1001)),'Original P curve lacks logical steps')
    require(all(int(r['cycle'])==order[int(r['step'])-1] for r in oldcurve),'Original P curve slot order differs')
    trainrefs={};gt_counts=[]
    for seq,cyc in SLOTS:
        count=0
        for idx in range(cyc*K+1,cyc*K+K):
            row=refs[seq,y7.A_ARM,idx]
            if helper.truth(row['gt_matched']):
                trainrefs[seq,idx]=dict(row);count+=1
        gt_counts.append(count)
    require(sum(gt_counts)==740 and sum(gt_counts[i] for i in order)==30834,'Training supervision/exposure differs')
    cache_plan={};extra_files={str(initpath):sha(initpath),str(curvepath.resolve()):sha(curvepath)}
    for seq in SEQUENCES:
        directory=Path(a.y04_dir).resolve()/'training/scene_caches'/seq/'cycle_cache'
        meta_path=directory/'cache_manifest.json';missing=[]
        if meta_path.is_file():
            meta=helper.json_read(meta_path)
            require(all(meta.get(k)==v for k,v in cache_signature(helper,base,seq).items()),f'Cache provenance differs: {seq}')
            extra_files[str(meta_path)]=sha(meta_path)
            for cyc in range(8):
                path=directory/f'cycle_{cyc:02d}.pt';entry=meta['cycles'].get(str(cyc),{})
                if not path.is_file():missing.append(cyc);continue
                require(entry.get('sha256')==sha(path),f'Old cache SHA differs: {path}')
                require(int(entry['supervised'])==gt_counts[SLOTS.index((seq,cyc))],'Cache supervised count differs')
                extra_files[str(path)]=entry['sha256']
        else:missing=list(range(8))
        cache_plan[seq]={'original_directory':str(directory),'reuse_all_eight':not missing,
            'missing_cycles':missing,'fallback':'build eight prefix cycles in new output only' if missing else 'read-only reuse'}
    for rel,expected in [('experiments/p1/gt4a2_c3_closed_loop.py',C3_SHA),
        ('experiments/p1/gt4a2_y04_fair_multiscene.py',Y04_SHA),
        ('experiments/p1/gt4a2_y07w_same_head_no_writeback.py',Y07_SHA)]:extra_files[str(ROOT/rel)]=expected
    pre={'status':'PREFLIGHT_PASS','version':VERSION,'script_sha256':sha(__file__),
        'initial_head_tensor_sha256':INIT_SHA,'initial_head_file':str(initpath),'sample_order_sha256':ORDER_SHA,
        'slots':[list(x) for x in SLOTS],'order':order,'unique_supervised_frames':740,
        'formal_committed_optimizer_updates':1000,'formal_committed_supervised_exposures':30834,
        'K':K,'tbptt':TBPTT,'lambda':LAMBDA,'seed':SEED,'learning_rate':1e-4,'weight_decay':1e-4,'clip_grad_norm':1.0,
        'mode':'pre_readout','gain':1.0,'new_training_arms':[ARM],
        'loss':'original robust log-depth + relu(current AbsRel - historical native A AbsRel), lambda=1',
        'training_rgb_ranges':{s:[0,255] for s in SEQUENCES},'partial_smoke':a.smoke_only,
        'cache_plan':cache_plan,'original_P_curve_rows':len(oldcurve),
        'original_P_duplicate_log_rows':len(oldcurve)-1000,'original_P_parameter_updates':1000,
        'y04a_dir':base['y04a_dir'],'y04_dir':base['y04_dir'],
        'input_sha256':{**base['input_sha256'],**extra_files},
        'limits':['Single training pilot, not a CVPR mechanism proof','Development tails, not independent scenes',
            'New checkpoint requires fresh feedback attribution if retained','No inference-time GT/A gate or new speed claim',
            'Committed schedule matches Y04; physical rollback work and actual compute are reported separately']}
    return pre,manifests,refs,trainrefs,rt,c3,y4,initial


def prepare_reader(a,helper,rt,c3,y4,pre,manifests,out,device):
    readers={};audits=[]
    for seq in SEQUENCES:
        plan=pre['cache_plan'][seq]
        if plan['reuse_all_eight']:directory=Path(plan['original_directory'])
        else:
            paths={'snapshot':Path(a.snapshot).resolve(),'crowd2':Path(getattr(a,seq+'_pair_csv')).resolve()}
            directory=c3.build_cycle_cache(paths,manifests[seq],out/'training/caches'/seq,device,limit=8)
        readers[seq]=c3.CycleReader(directory,max_items=1)
        meta=helper.json_read(directory/'cache_manifest.json')
        require(all(meta.get(k)==v for k,v in cache_signature(helper,{'manifests':{
            seq:{'sha256':pre['input_sha256'][str(Path(getattr(a,seq+'_pair_csv')).resolve())]}},
            'input_sha256':pre['input_sha256']},seq).items()),'Prepared cache signature differs')
        for cyc in range(8):
            obj=readers[seq].load(cyc);geometry=y4.validate_cycle_geometry(obj,f'Y08H/{seq}/{cyc}')
            for offset,frame in enumerate(obj['frames']):
                idx=cyc*K+offset;rec=manifests[seq][idx]
                require(int(frame['index'])==idx and frame['rgb_file']==Path(rec['rgb_path']).name,'Cache frame index/RGB differs')
                supervised=offset>0 and str(rec.get('gt_match_status','')).upper()=='MATCHED'
                require((frame['gt'] is not None)==supervised,'Cached GT assignment differs')
            audits.append({'sequence':seq,'cycle':cyc,'file':str(directory/f'cycle_{cyc:02d}.pt'),
                'sha256':meta['cycles'][str(cyc)]['sha256'],**geometry})
    helper.json_write(out/'y08h_cache_audit.json',audits)
    return y4.SceneCycleReader(readers,SLOTS),audits


def make_training_model(a,rt,initial,device):
    model=rt.gt2.load_model(Path(a.snapshot).resolve(),'pre_readout',device)
    head=model.pixel_decoder.persistent_feature_repair
    require(head is not None and model.pixel_decoder.feature_repair_mode=='pre_readout','Native P mode unavailable')
    head.load_state_dict(initial,strict=True)
    for p in model.parameters():p.requires_grad_(False)
    for p in head.parameters():p.requires_grad_(True)
    model.eval();head.train()
    require(sum(p.numel() for p in model.parameters() if p.requires_grad)==170496,'Training parameters differ')
    return model,head


def verify_A_training_targets(rt,c3,model,head,reader,trainrefs,device):
    """A zero-residual initial head must reproduce all matched frames in three cycles."""
    head.eval();records=[]
    with rt.torch.no_grad():
        for slot in (0,8,16):
            cycle=reader.load(slot);seq,cyc=SLOTS[slot]
            c3.warmup_shapes(model,SimpleNamespace(load=lambda _:cycle),device)
            radius,confidence,features=c3.move_state(cycle['keyframe_state'],device)
            rays=cycle['keyframe_state']['rays_map'].to(device)
            prev=c3.to_gpu_frame(cycle['frames'][0],device)
            local={'validity_mask':rt.torch.ones((1,1,*cycle['network_shape']),device=device,dtype=rt.torch.bool)}
            for off in range(1,32):
                frame=c3.to_gpu_frame(cycle['frames'][off],device)
                st=c3.to_gpu_static(cycle['statics'][off-1],device,local)
                with rt.torch.autocast(device_type='cuda',dtype=rt.torch.float16,enabled=True):
                    radius,confidence,features=rt.exp1.one_fast_step(model,prev,frame,st,radius,confidence,features)
                    if frame['gt'] is not None:
                        pred=rt.gt2.make_z_depth(radius,rays,cycle['padded_shape'],cycle['paddings'],model.interpolation_mode)
                        ref=training_reference(trainrefs,seq,cyc*32+off)
                        value,_=health_hinge(rt.torch,pred,frame['gt'],ref['AbsRel'])
                        require(abs(float(value.item())-float(ref['AbsRel']))<=2e-5,'Cached A AbsRel differs from historical training reference')
                        records.append({'sequence':seq,'frame_index':cyc*32+off,'A_AbsRel_deviation':abs(float(value.item())-float(ref['AbsRel']))})
                prev=frame
    head.train()
    return records


def train_cycle(rt,c3,model,head,optimizer,scaler,cycle,trainrefs,device,step):
    torch=rt.torch;seq=cycle['y04_sequence'];cyc=cycle['y04_local_cycle'];n=int(cycle['n_supervised'])
    radius,confidence,features=c3.move_state(cycle['keyframe_state'],device)
    rays=cycle['keyframe_state']['rays_map'].to(device)
    local={'validity_mask':torch.ones((1,1,*cycle['network_shape']),device=device,dtype=torch.bool)}
    prev=c3.to_gpu_frame(cycle['frames'][0],device)
    optimizer.zero_grad(set_to_none=True);frame_logs=[];segments=0
    for begin in range(1,32,TBPTT):
        terms=[];hold=[prev]
        for off in range(begin,min(32,begin+TBPTT)):
            curr=c3.to_gpu_frame(cycle['frames'][off],device)
            st=c3.to_gpu_static(cycle['statics'][off-1],device,local)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
                radius,confidence,features=rt.exp1.one_fast_step(model,prev,curr,st,radius,confidence,features)
                if curr['gt'] is not None:
                    idx=cyc*K+off;ref=training_reference(trainrefs,seq,idx)
                    pred=rt.gt2.make_z_depth(radius,rays,cycle['padded_shape'],cycle['paddings'],model.interpolation_mode)
                    absrel,hinge=health_hinge(torch,pred,curr['gt'],ref['AbsRel'])
                    base,count=rt.gt2.masked_log_depth_loss(pred,curr['gt'])
                    require(count==int(ref['valid_pixels']),'Training GT-valid count differs from A reference')
                    loss=base+LAMBDA*hinge
                    require(bool(torch.isfinite(loss.detach()).all().item()),'Nonfinite training loss')
                    terms.append(loss)
                    frame_logs.append({'step':step,'sequence':seq,'local_cycle':cyc,'frame_index':idx,'age':off,
                        'base_log_loss':float(base.detach().item()),'P_H_AbsRel':float(absrel.detach().item()),
                        'A_AbsRel':float(ref['AbsRel']),'hinge':float(hinge.detach().item()),
                        'penalty_active':float(hinge.detach().item())>0,'total_loss':float(loss.detach().item()),'valid_pixels':count})
            hold.append((curr,st));prev=curr
        if terms:
            scaler.scale(torch.stack(terms).sum()/float(n)).backward();segments+=1
        radius,confidence,features=c3.detach_state(radius,confidence,features)
        prev={key:(v.detach() if isinstance(v,torch.Tensor) else v) for key,v in prev.items()}
        del terms,hold
    require(len(frame_logs)==n,'Training GT exposure changed')
    scaler.unscale_(optimizer)
    gradients=[p.grad for p in head.parameters() if p.grad is not None]
    require(gradients and all(bool(torch.isfinite(g).all().item()) for g in gradients),'Missing/nonfinite gradients')
    require(any(int(torch.count_nonzero(g).item()) for g in gradients),'All gradients are zero')
    norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.0).item())
    old_scale=float(scaler.get_scale());scaler.step(optimizer);scaler.update()
    require(float(scaler.get_scale())>=old_scale,'GradScaler skipped an update; no silent budget change allowed')
    return {'step':step,'sequence':seq,'local_cycle':cyc,'slot':int(cycle['cycle']),
        'supervised_frames':n,'gradient_segments':segments,'gradient_norm':norm,
        'base_loss':statistics.mean(r['base_log_loss'] for r in frame_logs),
        'hinge':statistics.mean(r['hinge'] for r in frame_logs),
        'active_frames':sum(r['penalty_active'] for r in frame_logs),
        'total_loss':statistics.mean(r['total_loss'] for r in frame_logs)},frame_logs


def reconcile_recovery(obj,signature,order):
    require(obj['signature']==signature,'Recovery input/config identity differs')
    step=int(obj['completed_step']);curve=obj['curve'];frames=obj['frame_logs']
    require(0<step<=len(order) and [int(r['step']) for r in curve]==list(range(1,step+1)),'Recovery curve steps are duplicated/missing')
    require(all(int(r['slot'])==order[int(r['step'])-1] for r in curve),'Recovery slot order differs')
    require(len(frames)==sum(int(r['supervised_frames']) for r in curve),'Recovery exposure count differs')
    return step,curve,frames


def atomic_torch_save(torch,obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');torch.save(obj,tmp);tmp.replace(path)


def train(a,y7,helper,rt,c3,y4,initial,trainrefs,reader,pre,out,device):
    model,head=make_training_model(a,rt,initial,device)
    aligned=y4.GeometryAlignedReader(reader,model,'Y08H')
    warmup=c3.warmup_shapes(model,aligned,device)
    require(rt.exp1.state_sha256(c3.initial_state(head))==INIT_SHA,'Warmup modified initial head')
    guard=verify_A_training_targets(rt,c3,model,head,aligned,trainrefs,device)
    helper.csv_write(out/'y08h_training_A_checks.csv',guard)
    require(rt.exp1.state_sha256(c3.initial_state(head))==INIT_SHA,'A reference verification modified initialization')
    order=[0,8,16] if a.smoke_only else pre['order']
    signature=hashlib.sha256((json.dumps(pre,sort_keys=True,separators=(',',':'))+
        sha(out/'y08h_cache_audit.json')).encode()).hexdigest()
    train_dir=out/'training';train_dir.mkdir(parents=True,exist_ok=True)
    recovery=train_dir/'recovery.pt';final=train_dir/('P_H_smoke_step3.pt' if a.smoke_only else 'P_H_final_step1000.pt')
    optimizer=rt.torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4)
    scaler=rt.torch.amp.GradScaler('cuda',enabled=True)
    start=0;curve=[];frame_logs=[];resumes=[]
    if a.resume:
        require(recovery.is_file(),'No valid recovery checkpoint exists; start in a new directory')
        obj=rt.torch.load(recovery,map_location='cpu',weights_only=False)
        start,curve,frame_logs=reconcile_recovery(obj,signature,order)
        head.load_state_dict(obj['head'],strict=True);optimizer.load_state_dict(obj['optimizer']);scaler.load_state_dict(obj['scaler'])
        rt.torch.set_rng_state(obj['torch_rng']);rt.torch.cuda.set_rng_state(obj['cuda_rng'])
        rt.np.random.set_state(obj['numpy_rng']);random.setstate(obj['python_rng'])
        resume_path=train_dir/'resume_events.jsonl'
        with resume_path.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'restored_step':start,'time_unix':time.time()})+'\n')
        resumes=[json.loads(line) for line in resume_path.read_text().splitlines() if line.strip()]
        print(f'[RESUME] restored committed step {start}; logs are reconstructed from checkpoint, no duplicate rows',flush=True)
    else:
        require(not recovery.exists() and not final.exists(),'Existing training state; use --resume or a new output directory')
    eventpath=train_dir/'physical_update_events.jsonl'
    if not a.resume:require(not eventpath.exists(),'Existing update log; use a new output directory')
    rt.torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    try:
        for step in range(start+1,len(order)+1):
            cycle=aligned.load(order[step-1]);t=time.perf_counter()
            row,ff=train_cycle(rt,c3,model,head,optimizer,scaler,cycle,trainrefs,device,step)
            row['wall_sec']=time.perf_counter()-t;curve.append(row);frame_logs.extend(ff)
            with eventpath.open('a',encoding='utf-8') as f:f.write(json.dumps({'step':step,'supervised_frames':len(ff),'wall_sec':row['wall_sec']})+'\n')
            if step==1 or step%10==0 or step==len(order):
                print(f"[TRAIN H] {step}/{len(order)} {row['sequence']} cycle={row['local_cycle']} "
                    f"GT={row['supervised_frames']} base={row['base_loss']:.5f} hinge={row['hinge']:.5f} active={row['active_frames']}",flush=True)
            if step%25==0 or step==len(order):
                obj={'signature':signature,'completed_step':step,'head':c3.initial_state(head),
                    'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'curve':curve,'frame_logs':frame_logs,
                    'resume_events':resumes,'torch_rng':rt.torch.get_rng_state(),'cuda_rng':rt.torch.cuda.get_rng_state(),
                    'numpy_rng':rt.np.random.get_state(),'python_rng':random.getstate()}
                atomic_torch_save(rt.torch,obj,recovery)
                helper.csv_write(out/'y08h_training_curve.csv',curve)
        require(len(curve)==len(order) and [r['step'] for r in curve]==list(range(1,len(order)+1)),'Committed updates mismatch')
        if not a.smoke_only:require(len(frame_logs)==30834,'Committed GT exposures differ from Y04 P_multi')
        helper.csv_write(out/'y08h_training_curve.csv',curve)
        state=c3.initial_state(head)
        delta=math.sqrt(sum(float((state[k].float()-initial[k].float()).square().sum().item()) for k in state))
        require(delta>0 and all(bool(rt.torch.isfinite(v).all().item()) for v in state.values()),'Head did not update or became nonfinite')
        if final.exists():
            old=rt.torch.load(final,map_location='cpu',weights_only=False)
            require(old['signature']==signature and rt.exp1.state_sha256(old['repair_head_state_dict'])==rt.exp1.state_sha256(state),'Existing final head differs')
        else:
            atomic_torch_save(rt.torch,{'version':VERSION,'signature':signature,'mode':'pre_readout','gain':1.0,
                'steps':len(order),'initial_head_tensor_sha256':INIT_SHA,'sample_order_sha256':ORDER_SHA,
                'training_rgb_ranges':pre['training_rgb_ranges'],'lambda':LAMBDA,'repair_head_state_dict':state},final)
        loaded=rt.torch.load(final,map_location='cpu',weights_only=False)['repair_head_state_dict']
        require(rt.exp1.state_sha256(loaded)==rt.exp1.state_sha256(state),'Final checkpoint reload mismatch')
        events=[json.loads(line) for line in eventpath.read_text().splitlines() if line.strip()]
        summary={'status':'TRAINING_COMPLETE','steps':len(order),'committed_supervised_exposures':len(frame_logs),
            'initial_head_tensor_sha256':INIT_SHA,'final_head_tensor_sha256':rt.exp1.state_sha256(state),
            'checkpoint':str(final),'checkpoint_sha256':sha(final),'sample_order_sha256':ORDER_SHA,
            'param_delta_l2_from_init':delta,'reload':'PASS','warmup_keyframe_max_abs':warmup,
            'resumed':bool(resumes),'resume_events':resumes,'physical_completed_updates_logged':len(events),
            'physical_supervised_exposures_logged':sum(e['supervised_frames'] for e in events),
            'rolled_back_logged_updates':len(events)-len(order),
            'sum_logged_training_wall_sec':sum(e['wall_sec'] for e in events),
            'current_invocation_training_sec':time.perf_counter()-started,
            'peak_cuda_allocated_gib':rt.torch.cuda.max_memory_allocated(device)/1024**3,
            'compute_limit':'Log covers completed updates; interrupted in-flight work may be unlogged. Cache/warmup/checkpoint overhead is separate.'}
        helper.json_write(out/'y08h_training_summary.json',summary)
        helper.csv_write(out/'y08h_training_frames.csv',frame_logs)
        groups=defaultdict(list)
        for r in frame_logs:groups[r['sequence'],helper.age_bin(r['age'])].append(r)
        helper.csv_write(out/'y08h_penalty_activation.csv',[{'sequence':s,'age_bin':age,'frame_exposures':len(rr),
            'active_exposures':sum(r['penalty_active'] for r in rr),'active_fraction':statistics.mean(r['penalty_active'] for r in rr),
            'mean_hinge':statistics.mean(r['hinge'] for r in rr)} for (s,age),rr in sorted(groups.items())])
        return loaded,summary
    finally:
        del model,head,optimizer,scaler
        rt.torch.cuda.empty_cache()


@contextlib.contextmanager
def labels(module,**values):
    previous={key:getattr(module,key) for key in values}
    try:
        for key,value in values.items():setattr(module,key,value)
        yield
    finally:
        for key,value in previous.items():setattr(module,key,value)


def quality_tables(y7,helper,rows,refs):
    # Reuse the audited four-arm aggregation; only method labels are scoped.
    # Discard its N/P "feedback" tables: H/P compares objectives, not feedback.
    with labels(y7,N_ARM=ARM,ARMS=(*y7.REF_ARMS,ARM)):
        original=y7.build_tables(helper,rows,refs)
    suffixes=('all_frame_metrics','temporal_tail_metrics','sequence_summary','cycle_summary','age_summary','macro_summary')
    return {'y08h_'+suffix+'.csv':original['y07w_'+suffix+'.csv'] for suffix in suffixes}


def archive(out,pre,helper):
    dest=out/'y08h_review.zip';tmp=out/'y08h_review.zip.tmp'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for pattern in ('y08h_*.csv','y08h_*.json'):
            for p in sorted(out.glob(pattern)):
                if p.name!='y08h_failure.json':z.write(p,p.name)
        z.write(__file__,Path(__file__).name)
        for path in pre['input_sha256']:
            p=Path(path)
            if p.is_relative_to(ROOT/'experiments') or p.is_relative_to(ROOT/'velodepth'):
                if p.suffix=='.py':z.write(p,'source/'+str(p.relative_to(ROOT)))
        for name in ('y04a_summary.json','y04a_preflight.json','y04a_sequence_summary.csv'):
            z.write(Path(pre['y04a_dir'])/name,'reference/'+name)
    with zipfile.ZipFile(tmp) as z:require(z.testzip() is None,'Archive integrity failed')
    tmp.replace(dest);return dest


def main(argv=None):
    a=parse_args(argv)
    if a.self_test:return self_test()
    invocation_started=time.perf_counter()
    y7=load_module('y08h_y07','experiments/p1/gt4a2_y07w_same_head_no_writeback.py',Y07_SHA)
    helper=y7.import_helper()
    out=Path(a.out_dir).resolve();y04a=y7.resolve_y04a(a.y04a_dir,helper);a.y04a_dir=str(y04a)
    require(out!=y04a and y04a not in out.parents,'Output must be outside original Y04-A')
    with helper.exclusive_output(out,Path(a.y04_dir).resolve()):
        require(not (out/'y08h_summary.json').exists(),'Completed Y08-H output exists; choose a new directory')
        try:
            pre,manifests,refs,trainrefs,rt,c3,y4,initial=preflight(a,y7,helper)
            path=out/'y08h_preflight.json'
            if path.exists():require(helper.json_read(path)==pre,'Output identity differs; choose a new directory')
            elif a.resume:raise RuntimeError('Cannot resume without bound preflight')
            helper.json_write(path,pre)
            print(f'[PREFLIGHT_PASS] {VERSION}; lambda1; common init; 1000 committed steps; 30834 GT exposures',flush=True)
            if a.preflight_only:return 0
            require(rt.torch.cuda.is_available(),'CUDA is required')
            rt.torch.set_num_threads(4);rt.torch.manual_seed(SEED);rt.torch.cuda.manual_seed_all(SEED)
            rt.np.random.seed(SEED);random.seed(SEED)
            device=rt.torch.device('cuda')
            stage=time.perf_counter()
            reader,audits=prepare_reader(a,helper,rt,c3,y4,pre,manifests,out,device)
            cache_wall=time.perf_counter()-stage;stage=time.perf_counter()
            state,training=train(a,y7,helper,rt,c3,y4,initial,trainrefs,reader,pre,out,device)
            training_wall=time.perf_counter()-stage;stage=time.perf_counter()
            rows,checks,execution=[],[],[]
            with labels(helper,NEW_ARM=ARM):
                for seq in (['crowd2'] if a.smoke_only else SEQUENCES):
                    rr,cc,meta=helper.run_sequence(a,seq,manifests[seq],refs,rt,state,out)
                    rows.extend(rr);checks.extend(cc);execution.append(meta)
            evaluation_wall=time.perf_counter()-stage
            for source,expected in pre['input_sha256'].items():require(sha(source)==expected,f'Input changed: {source}')
            for entry in audits:require(sha(entry['file'])==entry['sha256'],'Cache changed during training')
            require(sha(__file__)==pre['script_sha256'],'Y08-H script changed during run')
            tables=quality_tables(y7,helper,rows,refs);tables['y08h_invariants.csv']=checks
            for name,rr in tables.items():helper.csv_write(out/name,rr)
            helper.json_write(out/'y08h_execution.json',execution)
            helper.json_write(out/'y08h_compute.json',{'timing_claim':False,'cache_prepare_wall_sec':cache_wall,
                'training_wrapper_wall_sec':training_wall,'evaluation_wall_sec':evaluation_wall,
                'invocation_wall_sec_before_archive':time.perf_counter()-invocation_started,
                'training_wrapper_includes':'model load, shape/A checks, optimizer work, recovery and checkpoint writes',
                'comparison_note':'Actual training cost differs after adding loss and checks; committed updates/exposure match Y04.'})
            summary={'status':'SMOKE_PASS' if a.smoke_only else 'MECHANICS_PASS','version':VERSION,
                'partial_smoke':a.smoke_only,'full_three_sequence_evaluation':not a.smoke_only,
                'training_performed':True,'new_training_arms':[ARM],'new_inference_arms':[ARM],
                'committed_optimizer_steps':training['steps'],'committed_supervised_exposures':training['committed_supervised_exposures'],
                'K':K,'lambda':LAMBDA,'mode':'pre_readout','gain':1.0,'tbptt':TBPTT,
                'checkpoint_sha256':training['checkpoint_sha256'],'initial_head_tensor_sha256':INIT_SHA,
                'final_head_tensor_sha256':training['final_head_tensor_sha256'],'sample_order_sha256':ORDER_SHA,
                'new_rgb_records':len(rows),'tail_comparison_records':len(tables['y08h_temporal_tail_metrics.csv']),
                'invariants_passed':True,'training_GT_scope':'three prefixes 0..255 only; tails not used by optimizer',
                'checkpoint_policy':'final committed update only; no validation selection',
                'script_sha256':sha(__file__),'preflight_sha256':sha(path),'timing_claim':False,
                'method_verdict':'NOT_AUTOMATICALLY_DECIDED; this is a single training-objective pilot','limits':pre['limits']}
            helper.json_write(out/'y08h_summary.json',summary)
            dest=archive(out,pre,helper);(out/'y08h_failure.json').unlink(missing_ok=True)
            for r in tables['y08h_sequence_summary.csv']:
                if r['arm']==ARM:print(f"[RESULT] {r['sequence']}: AbsRel={r['mean_AbsRel']:.6f}; vs P={r['relative_AbsRel_change_pct_vs_P_multi']:+.2f}%; harmed={r['harmed_frames_vs_A']}/{r['n_frames']}",flush=True)
            print(f"[{summary['status']}] {dest}",flush=True);return 0
        except Exception as exc:
            (out/'y08h_summary.json').unlink(missing_ok=True)
            helper.json_write(out/'y08h_failure.json',{'status':'FAILED','error':str(exc),'traceback':traceback.format_exc()})
            raise


def self_test():
    import unittest
    class Checks(unittest.TestCase):
        def test_locked_order(self):
            order=sample_order();self.assertEqual(len(order),1000)
            self.assertEqual(hashlib.sha256(','.join(map(str,order)).encode()).hexdigest(),ORDER_SHA)
        def test_tail_reference_rejected(self):
            with self.assertRaisesRegex(RuntimeError,'outside'):training_reference({},'balloon2',256)
            with self.assertRaisesRegex(RuntimeError,'outside'):training_reference({},'crowd2',0)
        def test_scoped_labels_restore_on_failure(self):
            m=SimpleNamespace(tag='old')
            with self.assertRaises(ValueError):
                with labels(m,tag='new'):raise ValueError('probe')
            self.assertEqual(m.tag,'old')
        def test_recovery_rejects_duplicate_steps(self):
            bad={'signature':'s','completed_step':2,'curve':[{'step':1,'slot':0,'supervised_frames':1}]*2,'frame_logs':[{},{}]}
            with self.assertRaisesRegex(RuntimeError,'duplicated'):reconcile_recovery(bad,'s',[0,1])
        def test_torch_hinge_gradient(self):
            try:import torch
            except ImportError:self.skipTest('torch unavailable locally; runs in server environment')
            gt=torch.ones(4);p=torch.full((4,),1.2,requires_grad=True)
            value,penalty=health_hinge(torch,p,gt,.1);penalty.backward()
            self.assertAlmostEqual(penalty.item(),.1,places=6)
            self.assertTrue(torch.all(p.grad>0).item())
            good=torch.full((4,),1.05,requires_grad=True)
            _,zero=health_hinge(torch,good,gt,.1);zero.backward()
            self.assertEqual(zero.item(),0);self.assertEqual(torch.count_nonzero(good.grad).item(),0)
            tie=torch.full((4,),1.25,requires_grad=True)
            _,zero=health_hinge(torch,tie,gt,.25);zero.backward()
            self.assertEqual(torch.count_nonzero(tie.grad).item(),0)
        def test_torch_invalid_prediction_not_masked(self):
            try:import torch
            except ImportError:self.skipTest('torch unavailable locally; runs in server environment')
            with self.assertRaisesRegex(RuntimeError,'Invalid prediction'):
                health_hinge(torch,torch.tensor([float('nan'),1.]),torch.ones(2),.1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:traceback.print_exc();raise SystemExit(1)
