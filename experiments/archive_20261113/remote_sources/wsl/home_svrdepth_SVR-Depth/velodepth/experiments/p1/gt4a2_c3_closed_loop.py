#!/usr/bin/env python3
"""GT4-A2-C3: true state-carrying training in 32-frame cycles.

Reuses the exact GT4-A2 Experiment1 model, DIS flow, preprocessing, loss,
and continuous evaluator. No architectural change, no prior C/D checkpoint
fine-tuning. Each optimizer step uses one complete 32-frame cycle; the model
parameters remain fixed throughout that cycle, with TBPTT every four
propagated frames. Residual state values are preserved across segments.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import cv2

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments'/'p1'))
import gt4a2_exp1_train_eval as exp1
import gt2_fixed_clip_repair_train as gt2

SEED=20261024
STEPS=1000
LR=1e-4
REFRESH=32
TBPTT=4
N_CYCLES=24
MODES={'C':'pre_readout','D':'post_readout'}
CACHE_FORMAT='c3-f32-context-v1'

def sha(p):
    return exp1.sha256_file(Path(p))

def write_json(p,obj):
    exp1.write_json(Path(p),obj)

def write_csv(p,rows):
    exp1.write_csv(Path(p),rows)

def config_args():
    ap=argparse.ArgumentParser(description='GT4-A2-C3 closed-loop state training')
    ap.add_argument('--snapshot',default=os.getenv('VDEEP_SNAPSHOT',''))
    ap.add_argument('--crowd2-pair-csv',default='outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv')
    ap.add_argument('--crowd3-pair-csv',default='outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv')
    ap.add_argument('--balloon2-pair-csv',default='outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv')
    ap.add_argument('--previous-config',default='outputs/gt4a2_exp1/gt4a2_exp1_config.json')
    ap.add_argument('--out-dir',default='outputs/gt4a2_c3')
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--smoke-only',action='store_true',help='one cycle, one update, no evaluation; no formal checkpoint')
    ap.add_argument('--resume',action='store_true',help='resume from recovery checkpoints without selecting intermediate weights')
    ap.add_argument('--steps',type=int,default=STEPS)
    return ap.parse_args()

def preflight(args):
    if not args.snapshot: raise RuntimeError('VDEEP_SNAPSHOT is unset')
    paths={k:Path(v).resolve() for k,v in {
        'snapshot':args.snapshot,'crowd2':args.crowd2_pair_csv,
        'crowd3':args.crowd3_pair_csv,'balloon2':args.balloon2_pair_csv,
        'old_config':args.previous_config}.items()}
    if not (paths['snapshot']/'model.safetensors').is_file():
        raise FileNotFoundError(paths['snapshot']/'model.safetensors')
    if not (paths['snapshot']/'config.json').is_file():
        raise FileNotFoundError(paths['snapshot']/'config.json')
    for k in ('crowd2','crowd3','balloon2','old_config'):
        if not paths[k].is_file(): raise FileNotFoundError(paths[k])
    old=json.loads(paths['old_config'].read_text(encoding='utf-8'))
    if old.get('protocol',{}).get('refresh_interval')!=REFRESH:
        raise RuntimeError('Earlier refresh interval differs')
    expected=old.get('source_hashes',{})
    if not expected: raise RuntimeError('Prior config source hashes missing')
    hashes={}
    for rel,digest in expected.items():
        p=ROOT/rel
        if not p.is_file():raise FileNotFoundError(p)
        actual=sha(p)
        hashes[rel]=actual
        if actual!=digest:
            raise RuntimeError(f'Original source has changed; refuse C3: {rel}: {actual} != {digest}')
    old_pairs=old.get('pair_hashes',{})
    for k in ('crowd2','crowd3','balloon2'):
        digest=sha(paths[k]); hashes[f'{k}_pairs']=digest
        if digest!=old_pairs.get(k):
            raise RuntimeError(f'{k} pair CSV hash differs from Experiment1; abort')
    if int(args.steps)!=STEPS and not args.smoke_only:
        raise RuntimeError('Formal C3 requires exactly 1000 optimizer steps per mode')
    rows=exp1.read_manifest(paths['crowd2'],training=False)
    if len(rows)<N_CYCLES*REFRESH:
        raise RuntimeError(f'Crowd2 needs at least {N_CYCLES*REFRESH} original RGB frames')
    anchor_rows=exp1.build_anchor_manifest(rows)
    if len(anchor_rows)!=N_CYCLES:
        raise RuntimeError('24-cycle anchor manifest not found')
    eval_rows={k:exp1.read_manifest(paths[k],training=False) for k in ('crowd3','balloon2')}
    for k,v in eval_rows.items():
        train_rgb={str(Path(r['rgb_path']).resolve()) for r in rows}
        train_gt={str(Path(r['depth_path']).resolve()) for r in rows if str(r.get('gt_match_status','')).upper()=='MATCHED'}
        test_rgb={str(Path(r['rgb_path']).resolve()) for r in v}
        test_gt={str(Path(r['depth_path']).resolve()) for r in v if str(r.get('gt_match_status','')).upper()=='MATCHED'}
        if (train_rgb & test_rgb) or (train_gt & test_gt):
            raise RuntimeError(f'Train/test physical file overlap: {k}')
    per_cycle=[]
    for cyc in range(N_CYCLES):
        start=cyc*REFRESH
        region=rows[start:start+REFRESH]
        matched=sum(str(r.get('gt_match_status','')).upper()=='MATCHED' for r in region[1:])
        if len(region)!=REFRESH or matched<1:
            raise RuntimeError(f'Cycle {cyc}: not 32 RGB frames or no supervised propagation')
        per_cycle.append({'cycle':cyc,'start':start,'end':start+31,'supervised_propagated_frames':matched,
                          'unsupervised_propagated_frames':31-matched})
    return paths,rows,per_cycle,{
        'preflight':'PASS','training_cycles':N_CYCLES,'training_rgb_indices':[0,767],
        'refresh':REFRESH,'tbptt_frames':TBPTT,'steps_per_mode':STEPS,'learning_rate':LR,
        'training_cycle_supervision':per_cycle,'training_pairs_sha256':hashes['crowd2_pairs'],
        'test_pair_hashes':{'crowd3':hashes['crowd3_pairs'],'balloon2':hashes['balloon2_pairs']},
        'original_source_hashes':{k:v for k,v in hashes.items() if not k.endswith('_pairs')},
        'original_config_sha256':sha(paths['old_config']),
        'pretrained_weights_sha256':sha(paths['snapshot']/'model.safetensors'),
        'dataset_rows':{'crowd2':len(rows),'crowd3':len(eval_rows['crowd3']),'balloon2':len(eval_rows['balloon2'])}}

def to_cpu_cache(frame, need_gt):
    result={
        'index':int(frame['index']),'img':frame['img'].detach().cpu().contiguous(),
        'gt':frame['gt'].detach().cpu().contiguous() if need_gt else None,
        'rgb_file':frame['rgb_file']}
    return result

def to_gpu_frame(frame,device):
    return {
        'index':int(frame['index']),
        'img':frame['img'].to(device,non_blocking=False),
        'gt':None if frame['gt'] is None else frame['gt'].to(device,non_blocking=False)}

def to_gpu_static(item, device, shape):
    return {
        'flow_norm':item['flow_norm'].to(device),
        'grid_sample':item['grid_sample'].to(device),
        'image_residual':item['image_residual'].to(device),
        'validity_mask':shape['validity_mask']}

def build_cycle_cache(paths,rows,out_dir,device,limit=None):
    cache_dir=out_dir/'cycle_cache'
    cache_dir.mkdir(parents=True,exist_ok=True)
    meta_path=cache_dir/'cache_manifest.json'
    signature={
        'format':CACHE_FORMAT,'pretrained_weights_sha256':sha(paths['snapshot']/'model.safetensors'),
        'pair_sha256':sha(paths['crowd2']),
        'original_source_hashes':{
            rel:sha(ROOT/rel) for rel in (
                'velodepth/models/decoder_velo.py',
                'velodepth/models/persistent_feature_repair.py',
                'experiments/p1/gt2_fixed_clip_repair_train.py',
                'experiments/p1/gt4a2_exp1_train_eval.py')},
    }
    n=N_CYCLES if limit is None else min(limit,N_CYCLES)
    reuse=False
    if meta_path.is_file():
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        reuse=all(meta.get(k)==v for k,v in signature.items())
        if not reuse:
            raise RuntimeError('Cycle cache provenance differs; do not reuse or overwrite old cache')
    else:
        meta=dict(signature);meta['cycles']={};write_json(meta_path,meta)
    missing=[]
    for cyc in range(n):
        p=cache_dir/f'cycle_{cyc:02d}.pt'
        entry=meta['cycles'].get(str(cyc),{})
        if not (p.is_file() and entry.get('sha256') and sha(p)==entry['sha256']):
            if p.is_file():
                raise RuntimeError(f'Cycle cache exists but hash is unknown/wrong: {p}')
            missing.append(cyc)
    if not missing:
        print(f'[CACHE] all {n} cycles already prepared, verified',flush=True)
        return cache_dir
    model=gt2.load_model(paths['snapshot'],'off',device)
    model.eval()
    shape=exp1.shape_info_for(model,Path(rows[0]['rgb_path']),device)
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    for cyc in missing:
        start=cyc*REFRESH
        fdata=[];statics=[]
        prev=None
        print(f'[CACHE] cycle={cyc+1}/{n}, original RGB indices={start}..{start+31}',flush=True)
        for offset in range(REFRESH):
            idx=start+offset
            status=str(rows[idx].get('gt_match_status','')).upper()
            need_gt=offset>0 and status=='MATCHED'
            current=exp1.load_frame(rows[idx],shape,device,need_gt=need_gt)
            fdata.append(to_cpu_cache(current,need_gt))
            if offset==0:
                with torch.no_grad():
                    rays,radius,conf,features=exp1.keyframe_state(model,current,shape)
                state={
                    'rays_map':rays.detach().cpu().clone(),
                    'radius':radius.detach().cpu().clone(),
                    'confidence':conf.detach().cpu().clone(),
                    'features':[x.detach().cpu().clone() for x in features]}
            else:
                with torch.no_grad():
                    st=exp1.transition_static(model,prev,current,shape,dis)
                statics.append({k:st[k].detach().cpu().contiguous() for k in (
                    'flow_norm','grid_sample','image_residual')})
            prev=current
            del current
        supervised=sum(f['gt'] is not None for f in fdata[1:])
        cache={'cycle':cyc,'frames':fdata,'statics':statics,'keyframe_state':state,
               'n_supervised':supervised,'network_shape':shape['network_shape'],
               'padded_shape':shape['padded_shape'],'paddings':shape['paddings'],
               'rgb_indices':[start,start+31]}
        if len(fdata)!=REFRESH or len(statics)!=31 or not 0<supervised<=31:
            raise RuntimeError(f'Invalid cache: cycle={cyc}')
        target=cache_dir/f'cycle_{cyc:02d}.pt'
        tmp=cache_dir/f'cycle_{cyc:02d}.tmp'
        torch.save(cache,tmp)
        os.replace(tmp,target)
        meta['cycles'][str(cyc)]={'sha256':sha(target),'bytes':target.stat().st_size,
                                   'supervised':supervised}
        write_json(meta_path,meta)
        del cache,fdata,statics,state,prev
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return cache_dir

class CycleReader:
    def __init__(self,cache_dir,max_items=2):
        self.dir=cache_dir;self.memo=OrderedDict();self.max_items=max_items
        self.manifest=json.loads((cache_dir/'cache_manifest.json').read_text(encoding='utf-8'))
        self.verified=set()
    def load(self,idx):
        if idx in self.memo:
            self.memo.move_to_end(idx)
            return self.memo[idx]
        p=self.dir/f'cycle_{idx:02d}.pt'
        expected=self.manifest['cycles'].get(str(idx),{}).get('sha256')
        if not expected:
            raise RuntimeError(f'Cached cycle checksum missing: {p}')
        if idx not in self.verified:
            if sha(p)!=expected:
                raise RuntimeError(f'Cached cycle checksum mismatch: {p}')
            self.verified.add(idx)
        obj=torch.load(p,map_location='cpu',weights_only=False)
        if obj['cycle']!=idx or len(obj['frames'])!=32 or len(obj['statics'])!=31:
            raise RuntimeError('Cycle cached structure mismatch')
        self.memo[idx]=obj
        if len(self.memo)>self.max_items:self.memo.popitem(last=False)
        return obj

def move_state(state,device):
    return (state['radius'].to(device).detach(),state['confidence'].to(device).detach(),
            [x.to(device).detach() for x in state['features']])

def detach_state(radius,confidence,features):
    return radius.detach(),confidence.detach(),[x.detach() for x in features]

def initial_state(seed_head):
    return {k:v.detach().cpu().clone() for k,v in seed_head.state_dict().items()}

def warmup_shapes(model, reader, device):
    cycle=reader.load(0)
    first=to_gpu_frame(cycle['frames'][0],device)
    dummy_shape={'validity_mask':torch.ones((1,1,*cycle['network_shape']),dtype=torch.bool,device=device),
                 'network_shape':cycle['network_shape']}
    with torch.no_grad():
        rays,radius,conf,features=exp1.keyframe_state(model,first,dummy_shape)
    cached=cycle['keyframe_state']
    comparisons=[(rays,cached['rays_map']),(radius,cached['radius']),
                 (conf,cached['confidence'])]
    comparisons += list(zip(features,cached['features']))
    differences=[(a.float().cpu()-b.float()).abs().max().item() for a,b in comparisons]
    maxdiff=max(differences)
    if maxdiff>1e-4:
        raise RuntimeError(f'Cache keyframe differs from current model: maxabs={maxdiff}')
    del rays,radius,conf,features
    return maxdiff

def train_cycle(model, head, optimizer, scaler, cycle, device):
    """Exactly 1 optimizer step per 32-frame cycle; each 4-frame TBPTT segment detaches."""
    if len(cycle['frames'])!=32 or len(cycle['statics'])!=31:
        raise RuntimeError('Cycle length mismatch')
    n_supervised=cycle['n_supervised']
    if n_supervised<1:raise RuntimeError('No GT in a training cycle')
    img0=cycle['frames'][0]
    radius,confidence,features=move_state(cycle['keyframe_state'],device)
    rays=cycle['keyframe_state']['rays_map'].to(device)
    pads=cycle['paddings'];pad_shape=cycle['padded_shape']
    validity=torch.ones((1,1,*cycle['network_shape']),device=device,dtype=torch.bool)
    local_shape={'validity_mask':validity}
    frame_losses=[];grad_segments=0;consumed=0;segment_lengths=[]
    optimizer.zero_grad(set_to_none=True)
    prev=to_gpu_frame(img0,device)
    for segment_begin in range(1,32,TBPTT):
        end=min(31,segment_begin+TBPTT-1)
        loss_terms=[]
        segment_hold=[prev]
        for offset in range(segment_begin,end+1):
            curr=to_gpu_frame(cycle['frames'][offset],device)
            static=to_gpu_static(cycle['statics'][offset-1],device,local_shape)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
                radius,confidence,features=exp1.one_fast_step(
                    model,prev,curr,static,radius,confidence,features)
                if curr['gt'] is not None:
                    pred=gt2.make_z_depth(radius,rays,pad_shape,pads,model.interpolation_mode)
                    loss,count=gt2.masked_log_depth_loss(pred,curr['gt'])
                    if not torch.isfinite(loss.detach()).all():
                        raise RuntimeError(f'Nonfinite loss at cycle={cycle["cycle"]} offset={offset}')
                    loss_terms.append(loss)
                    frame_losses.append(float(loss.detach().item()))
                    consumed+=1
            segment_hold.append((curr,static))
            prev=curr
        if loss_terms:
            segment_loss=torch.stack(loss_terms).sum()/float(n_supervised)
            scaler.scale(segment_loss).backward()
            grad_segments+=1
            del segment_loss
        radius,confidence,features=detach_state(radius,confidence,features)
        prev={k:(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in prev.items()}
        segment_lengths.append(end-segment_begin+1)
        del loss_terms,segment_hold
    if consumed!=n_supervised:
        raise RuntimeError(f'GT coverage mismatch: {consumed}!={n_supervised}')
    scaler.unscale_(optimizer)
    gradients=[p.grad for p in head.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise RuntimeError('No gradient or NaN/Inf gradients')
    nonzero=sum(bool(torch.count_nonzero(g).item()) for g in gradients)
    if not nonzero:raise RuntimeError('All repair-head gradients are zero')
    gradnorm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),max_norm=1.0).item())
    oldscale=float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    if float(scaler.get_scale())<oldscale:
        raise RuntimeError('GradScaler skipped optimizer update due to numerical overflow')
    return {'mean_log_depth_loss_before_update':float(np.mean(frame_losses)),
            'supervised_propagation_frames':consumed,'gradient_segments':grad_segments,
            'gradient_nonzero_tensors':nonzero,'gradient_norm_before_clip':gradnorm,
            'grad_scaler_scale':float(scaler.get_scale()),
            'segment_lengths':','.join(map(str,segment_lengths))}

def resume_states(out_dir,mode,optimizer,scaler,head,order_sha,init_sha,requested):
    path=out_dir/f'gt4a2_c3_{mode}_recovery.pt'
    if not path.is_file():return 0
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    if (ckpt['mode']!=mode or ckpt['sample_order_sha256']!=order_sha or
        ckpt['initial_head_state_sha256']!=init_sha or ckpt['steps_requested']!=requested):
        raise RuntimeError(f'Recovery checkpoint metadata mismatch: {path}')
    head.load_state_dict(ckpt['repair_head_state_dict'],strict=True)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    torch.set_rng_state(ckpt['torch_rng'])
    if torch.cuda.is_available():torch.cuda.set_rng_state(ckpt['cuda_rng'])
    np.random.set_state(ckpt['numpy_rng'])
    return int(ckpt['last_completed_step'])

def save_recovery(out_dir,mode,step,head,optimizer,scaler,order_sha,init_sha,requested):
    p=out_dir/f'gt4a2_c3_{mode}_recovery.pt'
    obj={'mode':mode,'last_completed_step':step,'steps_requested':requested,
         'sample_order_sha256':order_sha,'initial_head_state_sha256':init_sha,
         'repair_head_state_dict':initial_state(head),'optimizer_state_dict':optimizer.state_dict(),
         'scaler_state_dict':scaler.state_dict(),'torch_rng':torch.get_rng_state(),
         'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':np.random.get_state()}
    tmp=p.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,p)

def train_mode(label,model,reader,init_head_state,sample_order,out_dir,device,
               snapshot,pair_csv,steps,resume=False,smoke=False):
    print(f'[TRAIN {label}] loading model, closed-loop step = 32-frame cycle',flush=True)
    head=model.pixel_decoder.persistent_feature_repair
    if head is None:raise RuntimeError('Repair head absent')
    head.load_state_dict(init_head_state,strict=True)
    model.eval();head.train()
    params=[p for p in head.parameters() if p.requires_grad]
    if sum(p.numel() for p in params)!=170496 or sum(p.numel() for p in model.parameters() if p.requires_grad)!=170496:
        raise RuntimeError('Trainable parameter set differs from 170496 repair head only')
    maxdiff=warmup_shapes(model,reader,device)
    print(f'[TRAIN {label}] warm-up keyframe agrees with cache maxabs={maxdiff:g}',flush=True)
    optimizer=torch.optim.AdamW(params,lr=LR,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    order_sha=hashlib.sha256(','.join(map(str,sample_order)).encode()).hexdigest()
    init_sha=exp1.state_sha256(init_head_state)
    start_step=resume_states(out_dir,label,optimizer,scaler,head,order_sha,init_sha,steps) if resume else 0
    if start_step>steps:raise RuntimeError('Recovery step beyond target')
    curve_path=out_dir/f'gt4a2_c3_{label}_training_curve.csv'
    if start_step and not curve_path.is_file():raise RuntimeError('Resume training curve missing')
    if resume and not start_step and curve_path.is_file():
        raise RuntimeError('Training curve exists but no recoverable state: refuse to restart while resuming')
    if not start_step and curve_path.is_file() and not smoke:
        raise RuntimeError(f'Existing curve would be overwritten: {curve_path}. Use --resume or a new output dir.')
    start=time.perf_counter()
    curve=[]
    for step in range(start_step+1,steps+1):
        cyc=sample_order[step-1]
        c=reader.load(cyc)
        t0=time.perf_counter()
        metrics=train_cycle(model,head,optimizer,scaler,c,device)
        duration=time.perf_counter()-t0
        row={'mode':label,'step':step,'cycle':cyc,'cycle_start_index':cyc*32,
             'cycle_end_index':cyc*32+31,'optimizer_updates_this_cycle':1,
             'tbptt_frames':TBPTT,'wall_sec':duration,
             'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3,
             **metrics}
        curve.append(row)
        if step==1 or step%10==0 or step==steps:
            print(f'[{label}] step={step:04d}/{steps} cycle={cyc:02d} '
                  f'loss={row["mean_log_depth_loss_before_update"]:.6f} '
                  f'grad={row["gradient_norm_before_clip"]:.4g} '
                  f'segments={row["gradient_segments"]} '
                  f'vram={row["peak_cuda_allocated_gib"]:.2f}GiB',flush=True)
        if not smoke and (step%25==0 or step==steps):
            if start_step:
                # Append only completed steps, preserving previous records.
                with curve_path.open('r',encoding='utf-8-sig',newline='') as f:
                    old=list(csv.DictReader(f))
                curve=[*old,*curve]
                start_step=0
            write_csv(curve_path,curve)
        if not smoke and (step%100==0 or step==steps):
            save_recovery(out_dir,label,step,head,optimizer,scaler,order_sha,init_sha,steps)
    if smoke:
        print(f'[SMOKE {label}] one update completed; no formal checkpoint saved',flush=True)
        return {'smoke':True,'mode':label,'steps':steps,'last_loss':curve[-1]['mean_log_depth_loss_before_update']}
    head_dict=initial_state(head)
    checkpoint=out_dir/f'gt4a2_c3_{label}_final_step1000.pt'
    if checkpoint.exists():raise RuntimeError(f'Final checkpoint exists: {checkpoint}')
    meta={
        'experiment':'GT4-A2-C3 closed-loop training',
        'mode_label':label,'mode':MODES[label],'step_count':STEPS,
        'learning_rate':LR,'refresh_interval':REFRESH,
        'tbptt_frames':TBPTT,'optimizer_step_every':'one complete 32-frame cycle',
        'training_cycle_count':N_CYCLES,'training_original_rgb_indices':[0,767],
        'loss':'mean of valid-pixel robust log-depth losses across every matched propagation frame of the 32-frame cycle',
        'seed':SEED,'trainable_params':170496,'backbone_frozen':True,
        'sample_order_sha256':order_sha,
        'initial_head_state_sha256':init_sha,
        'repair_head_state_dict':head_dict,
        'pretrained_weights_sha256':sha(snapshot/'model.safetensors'),
        'training_pair_csv_sha256':sha(pair_csv),
        'checkpoint_policy':'last optimizer update, no validation selection'}
    torch.save(meta,checkpoint)
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if exp1.state_sha256(loaded['repair_head_state_dict'])!=exp1.state_sha256(head_dict):
        raise RuntimeError('Final checkpoint reload mismatch')
    summary={'mode':label,'steps':STEPS,'closed_loop_training':True,
             'trainable_parameters':170496,'elapsed_sec':time.perf_counter()-start,
             'initial_head_state_sha256':init_sha,'final_head_state_sha256':exp1.state_sha256(head_dict),
             'sample_order_sha256':order_sha,'checkpoint':str(checkpoint),
             'checkpoint_sha256':sha(checkpoint),'reload':'PASS',
             'mean_logged_train_loss':float(np.mean([float(r['mean_log_depth_loss_before_update']) for r in curve])),
             'mean_logged_step_wall_sec':float(np.mean([float(r['wall_sec']) for r in curve]))}
    write_json(out_dir/f'gt4a2_c3_{label}_training_summary.json',summary)
    print(f'[{label}] saved final checkpoint: {checkpoint}',flush=True)
    del optimizer,scaler
    return summary

def evaluate_all(out_dir,paths,device):
    frame_rows=[];eval_summaries=[];keyframe_checks=[]
    for seq in ('crowd2','crowd3','balloon2'):
        pair_csv=paths[seq]
        a_rows,a_summary,a_keyframes=exp1.evaluate_sequence_mode(
            seq,pair_csv,'A',None,None,paths['snapshot'],device)
        frame_rows.extend(a_rows);eval_summaries.append(a_summary)
        keyframe_checks.append({'sequence':seq,'mode':'A','max_diff_m':0.0,'pass':True})
        for label in ('C','D'):
            checkpoint=out_dir/f'gt4a2_c3_{label}_final_step1000.pt'
            rr,ss,_=exp1.evaluate_sequence_mode(
                seq,pair_csv,label,MODES[label],checkpoint,paths['snapshot'],device,
                a_keyframes=a_keyframes)
            frame_rows.extend(rr);eval_summaries.append(ss)
            diff=float(ss['keyframe_max_abs_depth_diff_vs_A_m'])
            keyframe_checks.append({'sequence':seq,'mode':label,'max_diff_m':diff,'pass':diff<=1e-4})
            if diff>1e-4:raise RuntimeError(f'Keyframe invariant failed: {seq}/{label} diff={diff}')
    write_csv(out_dir/'gt4a2_c3_full_sequence_frame_metrics.csv',frame_rows)
    write_csv(out_dir/'gt4a2_c3_keyframe_checks.csv',keyframe_checks)
    # Existing aggregator assumes exactly Crowd3 and Balloon2; formally use only these.
    heldout=[r for r in frame_rows if r['sequence'] in ('crowd3','balloon2')]
    heldout_summaries=[s for s in eval_summaries if s['sequence'] in ('crowd3','balloon2')]
    seq_rows,cycle_rows,decision=exp1.aggregate_results(heldout,heldout_summaries)
    write_csv(out_dir/'gt4a2_c3_sequence_summary.csv',seq_rows)
    write_csv(out_dir/'gt4a2_c3_cycle_summary.csv',cycle_rows)
    write_json(out_dir/'gt4a2_c3_decision.json',decision)
    crowd2={}
    for mode in ('A','C','D'):
        vals=[r for r in frame_rows if r['sequence']=='crowd2' and r['mode']==mode and r['AbsRel']!='']
        prop=[r for r in vals if r['is_keyframe']=='false']
        crowd2[mode]={'matched_frames':len(vals),
            'all_matched_frame_macro_metrics':{k:float(np.mean([float(r[k]) for r in vals]))
                for k in ('AbsRel','RMSE','delta1')},
            'propagated_only_macro_metrics':{k:float(np.mean([float(r[k]) for r in prop]))
                for k in ('AbsRel','RMSE','delta1')}}
    write_json(out_dir/'gt4a2_c3_crowd2_summary.json',crowd2)
    return eval_summaries,decision,crowd2,keyframe_checks

def report(out_dir,training,evals,decision,crowd2,pre):
    lines=['# GT4-A2-C3 连续状态训练闭环实验报告','',
           '**执行状态：PASS 表示流程完成；研究结论由预注册准入线单独判断。**','',
           '## 协议','',
           '- Crowd2 的 24 个完整 32 帧刷新周期作为训练样本；同一模型在每个周期内固定参数，连续执行最多 31 次修复。',
           '- 每4帧截断梯度并保留修复后的数值状态。每32帧完成一次 optimizer.step。',
           '- 每个有效 GT 传播帧等权贡献 log-depth loss；缺失 GT 的 RGB 仍参与状态传播。',
           '- C/D 共享初始化、优化顺序、AdamW lr=1e-4、各1000次更新、主干冻结。',
           '- 检查点选择：只使用最后第1000步；无测试集调参。',
           '- Crowd2、Crowd3、Balloon2 完整连续运行，每32 RGB索引刷新；不在每3帧重置。',
           '', '## 训练摘要', '',
           '|方法|优化步数|参数量|最终头SHA256|检查点重载|',
           '|---|---:|---:|---|---|']
    for mode in ('C','D'):
        v=training[mode]
        lines.append(f'|{mode}|{v["steps"]}|{v["trainable_parameters"]}|`{v["final_head_state_sha256"]}`|{v["reload"]}|')
    lines += ['', '## 3 条连续序列评价（每帧宏平均，包括关键帧）','',
              '|序列|分支|AbsRel ↓|RMSE (m) ↓|delta1 ↑|有效 GT 帧|',
              '|---|---|---:|---:|---:|---:|']
    for seq in ('crowd2','crowd3','balloon2'):
        for mode in ('A','C','D'):
            r=next(x for x in evals if x['sequence']==seq and x['mode']==mode)
            m=r['all_matched_frame_macro_metrics']
            lines.append(f'|{seq}|{mode}|{m["AbsRel"]:.6f}|{m["RMSE"]:.6f}|{m["delta1"]:.6f}|{r["matched_gt_frames_scored"]}|')
    lines += ['', '## 正式跨序列准入判定', '',
              '- 每条序列的 AbsRel 必须比 A 至少下降5%；RMSE 不恶化；delta1 下降不超过0.5个百分点。',
              f'- **结果：{decision["verdict"]}**',
              f'- 通过模式：{decision["passing_modes"]}',
              '- 仅评估最终检查点；所有刷新周期的胜负见 `gt4a2_c3_cycle_summary.csv`。',
              '- 本实验不启动原实验2，也不增加模块、误差图、刷新间隔扫描、其他随机种子。',
              '', '## 解释边界','',
              '- Crowd2 是训练序列，Crowd3/Balloon2 已参与开发决策，过关仅为项目准入，不代表最终独立泛化证明。',
              '- 本实验测试训练闭环是否能避免状态累积退化，不涉及新型结构的独立归因。',
              '']
    (out_dir/'GT4_A2_C3_report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    args=config_args()
    out_dir=Path(args.out_dir).resolve();out_dir.mkdir(parents=True,exist_ok=True)
    paths,rows,cycles,pre=preflight(args)
    write_json(out_dir/'gt4a2_c3_preflight.json',pre)
    print(f'[PASS] preflight: training cycles={len(cycles)}, eval sequences verified',flush=True)
    if args.preflight_only:return 0
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    final=out_dir/'gt4a2_c3_summary.json'
    if final.exists():raise RuntimeError('Formal completion marker exists; refuse overwrite')
    if not args.smoke_only and args.steps!=STEPS:raise RuntimeError('Formal steps must be 1000')
    if args.resume and not (out_dir/'gt4a2_c3_config.json').is_file():
        raise RuntimeError('Cannot resume: missing prior GT4-A2-C3 config')
    if args.smoke_only and args.steps!=1:raise RuntimeError('Smoke mode must use --steps 1')
    if args.smoke_only and 'gt4a2_c3_smoke' not in str(out_dir):
        raise RuntimeError('Smoke output must be isolated in a *_smoke directory')
    if args.resume and args.smoke_only:raise RuntimeError('Cannot resume a smoke test')
    torch.set_num_threads(4)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED)
    device=torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    signature={
        'experiment':'GT4-A2-C3 continuous-state closed-loop',
        'optimizer_updates_per_mode':STEPS,'tbptt_frames':TBPTT,
        'optimizer_update_interval':'once per 32-frame cycle',
        'learning_rate':LR,'weight_decay':1e-4,'grad_clip_norm':1.0,
        'training_cycles':24,'training_rgb_indices':[0,767],
        'source_sha256':{rel:sha(ROOT/rel) for rel in (
             'experiments/p1/gt4a2_c3_closed_loop.py',
             'experiments/p1/gt4a2_exp1_train_eval.py',
             'experiments/p1/gt2_fixed_clip_repair_train.py',
             'velodepth/models/decoder_velo.py',
             'velodepth/models/persistent_feature_repair.py')},
        'provenance':pre,'gpu':torch.cuda.get_device_name(0),
        'torch':torch.__version__,'python':platform.python_version(),
        'checkpoint_selection':'final step only','state_writes':'continuous C/D repair',
        'matched_gt':'valid-pixel log-depth loss, equal frame mean per cycle'}
    config_path=out_dir/'gt4a2_c3_config.json'
    if args.resume and config_path.is_file():
        prior=json.loads(config_path.read_text(encoding='utf-8'))
        if prior!=signature:raise RuntimeError('C3 configuration changed since recovery checkpoint was created')
    elif not args.resume:write_json(config_path,signature)
    cache_dir=build_cycle_cache(paths,rows,out_dir,device,limit=1 if args.smoke_only else None)
    reader=CycleReader(cache_dir,max_items=2)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    seed_model=gt2.load_model(paths['snapshot'],'pre_readout',device)
    head0=seed_model.pixel_decoder.persistent_feature_repair
    initial=initial_state(head0)
    init_sha=exp1.state_sha256(initial)
    if sum(p.numel() for p in head0.parameters())!=170496:
        raise RuntimeError('Repair head param count not 170496')
    del seed_model;torch.cuda.empty_cache()
    if not args.smoke_only and not args.resume:
        torch.save(initial,out_dir/'gt4a2_c3_initial_head.pt')
    order=exp1.generate_sample_order(24,STEPS,SEED)
    order_path=out_dir/'gt4a2_c3_sample_order.csv'
    if args.resume and not order_path.is_file():raise RuntimeError('Missing sample order on resume')
    if not args.resume and not args.smoke_only:
        write_csv(order_path,[{'step':i+1,'cycle':c,'cycle_start_index':32*c} for i,c in enumerate(order)])
    results={}
    for label in ('C','D'):
        if args.resume and (out_dir/f'gt4a2_c3_{label}_final_step1000.pt').exists():
            result=json.loads((out_dir/f'gt4a2_c3_{label}_training_summary.json').read_text(encoding='utf-8'))
            if sha(out_dir/f'gt4a2_c3_{label}_final_step1000.pt')!=result['checkpoint_sha256']:
                raise RuntimeError('Existing final checkpoint has wrong hash')
            results[label]=result
            print(f'[RESUME] {label} final checkpoint is already complete; skipping retraining',flush=True)
            continue
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        model=gt2.load_model(paths['snapshot'],MODES[label],device)
        # Smoke cache deliberately contains only cycle 0, so smoke must use
        # cycle 0 rather than the first random formal-training sample.
        # Formal training continues to use the unchanged fixed random order.
        smoke_order = [0] if args.smoke_only else order
        results[label]=train_mode(label,model,reader,initial,
                                   smoke_order,
                                   out_dir,device,paths['snapshot'],paths['crowd2'],
                                   args.steps,resume=args.resume,smoke=args.smoke_only)
        del model
        torch.cuda.empty_cache();gc.collect()
    if args.smoke_only:
        write_json(out_dir/'gt4a2_c3_smoke_summary.json',{'status':'PASS','result':results,
                   'both_modes_used_same_initialization':True,
                   'optimizer_updates_per_mode':1,'formal_training_started':False})
        print('[SMOKE] PASS C/D one full cycle optimizer update; no formal checkpoints/evaluation',flush=True)
        return 0
    if results['C']['initial_head_state_sha256']!=results['D']['initial_head_state_sha256'] or results['C']['initial_head_state_sha256']!=init_sha:
        raise RuntimeError('Different C/D initial repair head')
    print('[EVAL] Evaluating frozen C/D final checkpoints on all full sequences',flush=True)
    evals,decision,crowd2,keyframes=evaluate_all(out_dir,paths,device)
    summary={'status':'COMPLETE','experiment':'GT4-A2-C3 closed-loop state training',
             'training':results,'evaluation_summaries':evals,
             'crowd2':crowd2,'decision':decision,'keyframe_checks':keyframes,
             'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3,
             'config_sha256':sha(config_path)}
    write_json(final,summary)
    report(out_dir,results,evals,decision,crowd2,pre)
    print('[COMPLETE] Final report:',out_dir/'GT4_A2_C3_report.md',flush=True)
    print('[DECISION]',decision['verdict'],flush=True)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
