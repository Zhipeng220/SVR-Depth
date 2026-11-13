#!/usr/bin/env python3
"""Y04 fair multi-scene adaptation pilot for SVR-Depth (Fix2: padded-canvas readout alignment).

Core comparison (same 24 cycle slots, same 1000 optimizer updates, common initialization):
  P_single: persistent pre-readout repair trained on Crowd2 cycles 0..23
  O_single: output-only adapter trained on the same Crowd2 cycle schedule
  P_multi:  persistent pre-readout repair trained on 8 cycles each from Crowd2/Crowd3/Balloon2
  O_multi:  output-only adapter trained on exactly the same multi-scene schedule


Evaluation reports only temporal-tail frames that were not used for training:
  Crowd2 >= 768; Crowd3 >= 256; Balloon2 >= 256.
All three sequences remain development data, not independent blind test domains.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments' / 'p1'))
import gt4a2_q0_3_necessity_controls as q03
import gt4a2_c3_closed_loop as c3
import gt4a2_exp1_train_eval as exp1
import gt2_fixed_clip_repair_train as gt2

SEQUENCES = ('crowd2', 'crowd3', 'balloon2')
FORMAL_STEPS = 1000
SEED = 20261025
REFRESH = 32
TRAIN_CYCLES = {'crowd2': 24, 'crowd3': 8, 'balloon2': 8}
EVAL_START = {'crowd2': 768, 'crowd3': 256, 'balloon2': 256}


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows: Path(path).write_text('', encoding='utf-8'); return
    fields=list(dict.fromkeys(k for r in rows for k in r.keys()))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def args_for_q03(a):
    return SimpleNamespace(
        snapshot=a.snapshot, max_frames=0, crowd2_pair_csv=a.crowd2_pair_csv,
        crowd3_pair_csv=a.crowd3_pair_csv, balloon2_pair_csv=a.balloon2_pair_csv,
        previous_config=a.previous_config, c3_frame_csv=a.c3_frame_csv,
        c3_output_dir=a.c3_output_dir, c_checkpoint=a.c_checkpoint,
        d_checkpoint=a.d_checkpoint, metric_tolerance=1e-5,
        keyframe_pixel_tolerance_m=1e-3, sequences=','.join(SEQUENCES),
        smoke_only=False, resume=False, out_dir=a.out_dir, preflight_only=True)

class SceneCycleReader:
    """Map a stable slot index to a (sequence, local-cycle) cache entry."""
    def __init__(self, readers, slots): self.readers=readers; self.slots=list(slots)
    def load(self, slot):
        if slot < 0 or slot >= len(self.slots): raise IndexError(f'bad cycle slot {slot}')
        seq, cyc = self.slots[slot]
        obj=self.readers[seq].load(cyc)
        out=dict(obj); out['cycle']=int(slot); out['y04_sequence']=seq; out['y04_local_cycle']=int(cyc)
        return out

def balanced_order(n_slots, steps, seed):
    rng=random.Random(seed); order=[]
    while len(order)<steps:
        block=list(range(n_slots)); rng.shuffle(block); order.extend(block)
    return order[:steps]

def make_q03_args(a):
    return args_for_q03(a)

def build_caches(a, context, manifests, out_dir, device, smoke):
    cache_root=out_dir/'training'/'scene_caches'; cache_root.mkdir(parents=True, exist_ok=True)
    readers={}; cache_meta={}
    for seq in SEQUENCES:
        n=1 if smoke else TRAIN_CYCLES[seq]
        rows=manifests[seq]
        if len(rows) < n*REFRESH: raise RuntimeError(f'{seq}: need {n*REFRESH} frames, have {len(rows)}')
        # The audited C3 cache builder uses paths['crowd2'] as the pair-manifest signature.
        paths=dict(context['paths']); paths['crowd2']=Path(getattr(a, f'{seq}_pair_csv')).resolve()
        cache_out=cache_root/seq
        cache_dir=c3.build_cycle_cache(paths, rows, cache_out, device, limit=n)
        readers[seq]=c3.CycleReader(cache_dir, max_items=2)
        entries=[]
        for cyc in range(n):
            cycle=readers[seq].load(cyc)
            entries.append({'sequence':seq,'cycle':cyc,'rgb_start':cyc*REFRESH,
                            'rgb_end':cyc*REFRESH+REFRESH-1,
                            'supervised_gt_frames':int(cycle['n_supervised'])})
        cache_meta[seq]=entries
    return readers, cache_meta

def write_checkpoint_provenance(training_dir, label, kind, sequences, slots, order, init_hash, snapshot_hash, gt_frames, artifact_label):
    """Attach the explicit Y04 split provenance to each final checkpoint and summary."""
    training_dir=Path(training_dir)
    if kind=='persistent':
        cp=training_dir/'gt4a2_c3_C_final_step1000.pt'
        summ=training_dir/'gt4a2_c3_C_training_summary.json'
        if not cp.is_file(): raise FileNotFoundError(cp)
        obj=torch.load(cp,map_location='cpu',weights_only=False)
        obj.update({'experiment':'Y04 fair multi-scene adaptation pilot','y04_label':artifact_label,'y04_kind':kind,
                    'training_sequences':list(sequences),'training_cycle_slots':[list(x) for x in slots],
                    'optimizer_steps':FORMAL_STEPS,'unique_gt_frames_across_cycle_slots':int(gt_frames),
                    'y04_sample_order_sha256':hashlib.sha256(','.join(map(str,order)).encode()).hexdigest(),
                    'y04_initial_head_sha256':init_hash,'y04_pretrained_weights_sha256':snapshot_hash,
                    'selection_policy':'final step only; no validation-based checkpoint selection'})
        torch.save(obj,cp)
        if summ.is_file():
            s=json.loads(summ.read_text(encoding='utf-8')); s['checkpoint_sha256']=sha256_file(cp); s['y04_training_sequences']=list(sequences); write_json(summ,s)
    else:
        cp=training_dir/f'{label}_final_step{FORMAL_STEPS}.pt'
        summ=training_dir/f'{label}_training_summary.json'
        if not cp.is_file(): raise FileNotFoundError(cp)
        obj=torch.load(cp,map_location='cpu',weights_only=False)
        obj.update({'experiment':'Y04 fair multi-scene adaptation pilot','y04_label':artifact_label,'y04_kind':kind,
                    'training_sequences':list(sequences),'training_cycle_slots':[list(x) for x in slots],
                    'optimizer_steps':FORMAL_STEPS,'unique_gt_frames_across_cycle_slots':int(gt_frames),
                    'y04_sample_order_sha256':hashlib.sha256(','.join(map(str,order)).encode()).hexdigest(),
                    'y04_initial_head_sha256':init_hash,'y04_pretrained_weights_sha256':snapshot_hash,
                    'selection_policy':'final step only; no validation-based checkpoint selection'})
        torch.save(obj,cp)
        if summ.is_file():
            s=json.loads(summ.read_text(encoding='utf-8')); s['checkpoint_sha256']=sha256_file(cp); s['y04_training_sequences']=list(sequences); write_json(summ,s)

def train_persistent(a, label, slots, reader, order, init_state, out_dir, device, smoke, steps):
    mode='pre_readout'
    train_dir=out_dir/'training'/label
    final_cp=train_dir/'gt4a2_c3_C_final_step1000.pt'
    final_summary=train_dir/'gt4a2_c3_C_training_summary.json'
    if not smoke and a.resume and final_cp.is_file() and final_summary.is_file():
        s=json.loads(final_summary.read_text(encoding='utf-8'))
        if s.get('checkpoint_sha256') != sha256_file(final_cp): raise RuntimeError(f'Completed checkpoint hash mismatch: {final_cp}')
        print(f'[RESUME] completed persistent arm {label}; verified checkpoint',flush=True)
        return s
    model=gt2.load_model(Path(a.snapshot).resolve(), mode, device)
    try:
        result=c3.train_mode('C', model, reader, init_state, order,
            train_dir, device, Path(a.snapshot).resolve(),
            Path(a.crowd2_pair_csv).resolve(), steps, resume=a.resume and not smoke, smoke=smoke)
    finally:
        del model; torch.cuda.empty_cache(); gc.collect()
    return result

def train_output(a, label, reader, order, init_state, out_dir, device, smoke, steps):
    train_dir=out_dir/'training'/label
    final_cp=train_dir/f'{label}_final_step1000.pt'
    final_summary=train_dir/f'{label}_training_summary.json'
    if not smoke and a.resume and final_cp.is_file() and final_summary.is_file():
        s=json.loads(final_summary.read_text(encoding='utf-8'))
        if s.get('checkpoint_sha256') != sha256_file(final_cp): raise RuntimeError(f'Completed checkpoint hash mismatch: {final_cp}')
        print(f'[RESUME] completed output-only arm {label}; verified checkpoint',flush=True)
        return s
    model, module=q03.load_base_model(Path(a.snapshot).resolve(), device)
    label_full=label
    # Q0-3's output-only helper enters the fast decoder path from a cached state.
    # That path does not initialize RadialModule.original_shapes as native
    # keyframe inference does. The decoder output must remain on the padded
    # network canvas so it aligns with the cached radius/rays_map used by
    # make_z_depth(); do NOT subtract letterbox paddings here.
    original_load = reader.load
    def load_with_original_shapes(slot):
        cycle = original_load(slot)
        padded_shape = tuple(int(x) for x in cycle["padded_shape"])
        if len(padded_shape) != 2 or min(padded_shape) <= 0:
            raise RuntimeError(f"{label}: invalid padded_shape in cycle cache: {padded_shape}")
        state_rays = cycle["keyframe_state"]["rays_map"]
        ray_hw = tuple(int(x) for x in state_rays.shape[-2:])
        if ray_hw != padded_shape:
            raise RuntimeError(
                f"{label}: cache geometry mismatch: padded_shape={padded_shape}, "
                f"keyframe rays_map HW={ray_hw}"
            )
        model.pixel_decoder.radial_module.set_original_shapes(padded_shape)
        return cycle
    reader.load = load_with_original_shapes
    try:
        result=q03.train_one_output_only_module(label_full, model, module, reader, order,
            train_dir, steps=steps, initial_state=init_state, lr=q03.LR,
            smoke=smoke, resume=a.resume and not smoke)
    finally:
        reader.load = original_load
        del model, module; torch.cuda.empty_cache(); gc.collect()
    return result

def metric_summary(frame_rows, split_rows):
    # Keep only matched GT rows within locked temporal-tail split.
    cutoffs=EVAL_START
    valid=[]
    for r in frame_rows:
        seq=str(r.get('sequence','')); idx=int(r.get('frame_index_zero_based', r.get('frame_index', -1)))
        if seq not in cutoffs or idx < cutoffs[seq]: continue
        if not str(r.get('gt_matched','true')).lower() in ('true','1','yes'): continue
        valid.append(r)
    summary=[]
    for seq in SEQUENCES:
        for arm in sorted({str(r.get('arm','')) for r in valid}):
            group=[r for r in valid if r.get('sequence')==seq and r.get('arm')==arm]
            if not group: continue
            row={'sequence':seq,'arm':arm,'n_frames':len(group),'eval_start_index':cutoffs[seq]}
            for key in ('AbsRel','RMSE','delta1'):
                vals=[]
                for r in group:
                    try: v=float(r.get(key,''))
                    except (TypeError,ValueError): continue
                    if np.isfinite(v): vals.append(v)
                if vals: row[f'mean_{key}']=float(np.mean(vals))
            summary.append(row)
    return valid, summary

def main():
    p=argparse.ArgumentParser(description='Y04 fair multi-scene adaptation pilot')
    p.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT',''))
    p.add_argument('--crowd2-pair-csv',default='outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv')
    p.add_argument('--crowd3-pair-csv',default='outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv')
    p.add_argument('--balloon2-pair-csv',default='outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv')
    p.add_argument('--previous-config',default='outputs/gt4a2_exp1/gt4a2_exp1_config.json')
    p.add_argument('--c3-frame-csv',default='outputs/gt4a2_c3/gt4a2_c3_full_sequence_frame_metrics.csv')
    p.add_argument('--c3-output-dir',default='outputs/gt4a2_c3')
    p.add_argument('--c-checkpoint',default='outputs/gt4a2_c3/gt4a2_c3_C_final_step1000.pt')
    p.add_argument('--d-checkpoint',default='outputs/gt4a2_c3/gt4a2_c3_D_final_step1000.pt')
    p.add_argument('--out-dir',default='outputs/gt4a2_y04_validation')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--steps',type=int,default=FORMAL_STEPS)
    a=p.parse_args()
    for seq in SEQUENCES: setattr(a,f'{seq}_pair_csv',getattr(a,f'{seq}_pair_csv'))
    if not a.snapshot: raise RuntimeError('VDEEP_SNAPSHOT is unset; pass --snapshot or export it')
    if not a.smoke_only and a.steps!=FORMAL_STEPS: raise ValueError('Formal Y04 requires 1000 optimizer updates per arm')
    out_dir=Path(a.out_dir).resolve()
    marker=out_dir/'y04_summary.json'
    if marker.exists() and not a.preflight_only: raise RuntimeError(f'Y04 completion marker exists; use a new --out-dir: {marker}')
    qa=args_for_q03(a)
    pre, manifests, historical_refs, sample_order, order_sha, context=q03.build_preflight(qa,list(SEQUENCES))
    # Training split: 24 slots total in both conditions; 8 cycles per scene in the multi-scene arm.
    single_slots=[('crowd2',i) for i in range(24)]
    multi_slots=[('crowd2',i) for i in range(8)]+[('crowd3',i) for i in range(8)]+[('balloon2',i) for i in range(8)]
    split=[]
    for seq, rows in manifests.items():
        train_end=TRAIN_CYCLES[seq]*REFRESH if seq in TRAIN_CYCLES else 0
        for idx, rec in enumerate(rows):
            split_name='train_fit' if idx < train_end else 'temporal_tail_eval'
            split.append({'sequence':seq,'frame_index_zero_based':idx,'rgb_file':Path(rec['rgb_path']).name,
                          'gt_file':Path(rec['depth_path']).name if str(rec.get('gt_match_status','')).upper()=='MATCHED' else '',
                          'gt_matched':str(str(rec.get('gt_match_status','')).upper()=='MATCHED').lower(),
                          'multi_scene_split':split_name,'used_for_fit_multi_scene':str(idx<train_end).lower(),
                          'single_scene_split':'train_fit' if seq=='crowd2' and idx<768 else 'temporal_or_cross_scene_eval',
                          'used_for_fit_single_scene':str(seq=='crowd2' and idx<768).lower()})
    protocol={'status':'PREFLIGHT_PASS','experiment':'Y04 fair multi-scene adaptation pilot',
      'snapshot':str(Path(a.snapshot).resolve()),'pretrained_weights_sha256':sha256_file(Path(a.snapshot)/'model.safetensors'),
      'sequences':list(SEQUENCES),'refresh_interval':REFRESH,'formal_steps_per_arm':FORMAL_STEPS,
      'common_initial_head_sha256':context['initial_head_hash'],
      'single_scene_slots':[list(x) for x in single_slots],'multi_scene_slots':[list(x) for x in multi_slots],
      'slot_budget':'24 distinct 32-frame cycles per condition; same 1000 optimizer updates and same slot-order schedule for persistent/output-only within condition',
      'single_scene_fit':'Crowd2 RGB/GT indices 0..767 (24 cycles)',
      'multi_scene_fit':'Crowd2 0..255 + Crowd3 0..255 + Balloon2 0..255 (8 cycles per sequence)',
      'temporal_tail_eval_start':EVAL_START,
      'evaluation_note':'All sequences are development data; tail split is temporal holdout, not an independent blind test domain.',
      'primary_arms':['A_native','Y04_O_single','Y04_P_single_gain0.5','Y04_O_multi','Y04_P_multi_gain0.5'],
      'checkpoint_selection':'final step only; no held-out selection',
      'historical_q03_preflight':pre}
    out_dir.mkdir(parents=True,exist_ok=True); write_json(out_dir/'y04_preflight.json',protocol); write_csv(out_dir/'y04_split_manifest.csv',split)
    print('[PASS] Y04 preflight: 24 matched cycle slots per condition; common initialization; temporal tails locked',flush=True)
    if a.preflight_only:
        print('Y04 preflight complete; no training or inference performed.'); return 0
    if not torch.cuda.is_available(): raise RuntimeError('Y04 training/evaluation requires CUDA')
    torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    device=torch.device('cuda'); os.environ['VDEEP_SNAPSHOT']=str(Path(a.snapshot).resolve())
    readers, cache_meta=build_caches(a,context,manifests,out_dir,device,a.smoke_only)
    write_json(out_dir/'y04_training_cycle_manifest.json',cache_meta)
    init_state=context['initial_head_state']
    steps=3 if a.smoke_only else FORMAL_STEPS
    single_reader=SceneCycleReader(readers, [('crowd2',i) for i in range(1 if a.smoke_only else 24)])
    multi_reader=SceneCycleReader(readers, multi_slots if not a.smoke_only else [('crowd2',0),('crowd3',0),('balloon2',0)])
    single_order=[0] if a.smoke_only else balanced_order(24,steps,SEED)
    multi_order=[0,1,2] if a.smoke_only else list(single_order)
    actual_single_gt=sum(int(readers['crowd2'].load(i)['n_supervised']) for i in range(1 if a.smoke_only else 24))
    actual_multi_gt=sum(int(readers[s].load(c)['n_supervised']) for s,c in (multi_slots if not a.smoke_only else [('crowd2',0),('crowd3',0),('balloon2',0)]))
    write_json(out_dir/'y04_budget_manifest.json',{'single_scene_unique_gt_frames_across_slots':actual_single_gt,
      'multi_scene_unique_gt_frames_across_slots':actual_multi_gt,'single_scene_optimizer_steps':steps,
      'multi_scene_optimizer_steps':steps,'single_scene_slots':len(single_reader.slots),
      'multi_scene_slots':len(multi_reader.slots),'gt_exposure_difference_must_be_reported':actual_single_gt!=actual_multi_gt})
    training_results={}
    if a.smoke_only:
        # Smoke exercises all three scenes and both training pathways with 1/3/1/3 optimizer updates.
        training_results['P_single']=train_persistent(a,'P_single',single_reader.slots,single_reader,single_order,init_state,out_dir,device,True,1)
        training_results['O_single']=train_output(a,'O_single',single_reader,single_order,init_state,out_dir,device,True,1)
        training_results['P_multi']=train_persistent(a,'P_multi',multi_reader.slots,multi_reader,multi_order,init_state,out_dir,device,True,3)
        training_results['O_multi']=train_output(a,'O_multi',multi_reader,multi_order,init_state,out_dir,device,True,3)
    else:
        # Training arm order is fixed and recorded; each arm starts from the same head initialization.
        training_results['P_single']=train_persistent(a,'P_single',single_reader.slots,single_reader,single_order,init_state,out_dir,device,False,steps)
        training_results['O_single']=train_output(a,'O_single',single_reader,single_order,init_state,out_dir,device,False,steps)
        training_results['P_multi']=train_persistent(a,'P_multi',multi_reader.slots,multi_reader,multi_order,init_state,out_dir,device,False,steps)
        training_results['O_multi']=train_output(a,'O_multi',multi_reader,multi_order,init_state,out_dir,device,False,steps)
        init_hash=exp1.state_sha256(init_state); snap_hash=sha256_file(Path(a.snapshot)/'model.safetensors')
        write_checkpoint_provenance(out_dir/'training'/'P_single','C','persistent',['crowd2'],single_reader.slots,single_order,init_hash,snap_hash,actual_single_gt,'Y04_P_single')
        write_checkpoint_provenance(out_dir/'training'/'O_single','Y04_O_single','output_only',['crowd2'],single_reader.slots,single_order,init_hash,snap_hash,actual_single_gt,'Y04_O_single')
        write_checkpoint_provenance(out_dir/'training'/'P_multi','C','persistent',list(SEQUENCES),multi_reader.slots,multi_order,init_hash,snap_hash,actual_multi_gt,'Y04_P_multi')
        write_checkpoint_provenance(out_dir/'training'/'O_multi','Y04_O_multi','output_only',list(SEQUENCES),multi_reader.slots,multi_order,init_hash,snap_hash,actual_multi_gt,'Y04_O_multi')
        single_gt_per_slot=[int(readers['crowd2'].load(i)['n_supervised']) for i in range(24)]
        multi_gt_per_slot=[int(readers[s].load(c)['n_supervised']) for s,c in multi_slots]
        single_exposure=sum(single_gt_per_slot[i] for i in single_order)
        multi_exposure=sum(multi_gt_per_slot[i] for i in multi_order)
        # Refresh final checkpoint hashes after attaching Y04 provenance.
        checkpoint_paths={
          'P_single':out_dir/'training'/'P_single'/'gt4a2_c3_C_final_step1000.pt',
          'O_single':out_dir/'training'/'O_single'/'Y04_O_single_final_step1000.pt',
          'P_multi':out_dir/'training'/'P_multi'/'gt4a2_c3_C_final_step1000.pt',
          'O_multi':out_dir/'training'/'O_multi'/'Y04_O_multi_final_step1000.pt'}
        for name, cp in checkpoint_paths.items():
            if name in training_results and cp.is_file(): training_results[name]['checkpoint_sha256']=sha256_file(cp)
        write_json(out_dir/'y04_training_compute.json', {'status':'TRAINING_COMPLETE','steps_per_arm':steps,
          'training_results':training_results,'unique_cycle_gt_frames':{'single_scene':actual_single_gt,'multi_scene':actual_multi_gt},
          'sampled_gt_frame_exposures':{'single_scene':single_exposure,'multi_scene':multi_exposure},
          'gt_exposure_difference':multi_exposure-single_exposure,'interpretation':'Equal optimizer updates and cycle-slot count; report actual GT exposure difference rather than assuming exact equality.'})
    del readers, single_reader, multi_reader; torch.cuda.empty_cache(); gc.collect()
    if a.smoke_only:
        write_json(marker,{'status':'SMOKE_PASS','training_performed':True,'formal_checkpoints_saved':False,
                           'steps':steps,'sequences':list(SEQUENCES),'note':'Smoke is mechanics only; do not interpret metrics.'})
        print(f'[SMOKE PASS] Y04 mechanics package: {out_dir}',flush=True); return 0
    # Load trained modules and evaluate matched temporal tails. Run all arms from sequence start, score tails only.
    snap=Path(a.snapshot).resolve(); train_dir=out_dir/'training'; device=torch.device('cuda')
    ck_p_single=torch.load(train_dir/'P_single'/'gt4a2_c3_C_final_step1000.pt',map_location='cpu',weights_only=False)
    ck_p_multi=torch.load(train_dir/'P_multi'/'gt4a2_c3_C_final_step1000.pt',map_location='cpu',weights_only=False)
    ck_o_single=torch.load(train_dir/'O_single'/'Y04_O_single_final_step1000.pt',map_location='cpu',weights_only=False)
    ck_o_multi=torch.load(train_dir/'O_multi'/'Y04_O_multi_final_step1000.pt',map_location='cpu',weights_only=False)
    p_single=q03.make_frozen_head(ck_p_single['repair_head_state_dict'],device)
    p_multi=q03.make_frozen_head(ck_p_multi['repair_head_state_dict'],device)
    # Build adapter instances using the registered repair-head architecture from the shared initial state.
    _, o_single=q03.load_base_model(snap,device); o_single.load_state_dict(ck_o_single['adapter_state_dict'],strict=True); o_single.eval()
    _, o_multi=q03.load_base_model(snap,device); o_multi.load_state_dict(ck_o_multi['adapter_state_dict'],strict=True); o_multi.eval()
    all_rows=[]; run_meta=[]
    eval_arms=[('A_native',None,'native'),('Y04_O_single',o_single,'output'),('Y04_P_single_gain0.5',p_single,'persistent'),
               ('Y04_O_multi',o_multi,'output'),('Y04_P_multi_gain0.5',p_multi,'persistent')]
    for seq in SEQUENCES:
        rows=manifests[seq]
        if a.smoke_only: rows=rows[:64]
        model,_tmp=q03.load_base_model(snap,device)
        shape=exp1.shape_info_for(model,Path(rows[0]['rgb_path']),device)
        for label,adapter,kind in eval_arms:
            if kind=='native': arm='A_native'; schedule='none'; gain=1.0; used_adapter=None
            elif kind=='output': arm='B3_output_adapter_trained_output_only'; schedule='none'; gain=1.0; used_adapter=adapter
            else: arm='S2_C_gain0.5_R1_F_only'; schedule='every_prop'; gain=0.5; used_adapter=adapter
            fr,meta,_keys=q03.run_arm(seq,arm,model,rows,shape,device,adapter=used_adapter,gain=gain,schedule=schedule)
            for r in fr:
                r['arm']=label
                r['y04_eval_start_index']=EVAL_START[seq]
                r['y04_split']='smoke_all_frames' if a.smoke_only else ('temporal_tail_eval' if int(r.get('frame_index_zero_based',r.get('frame_index',-1)))>=EVAL_START[seq] else 'train_prefix_not_scored')
            all_rows.extend(fr); meta['y04_arm']=label; run_meta.append(meta)
            print(f'[EVAL] {seq}/{label} done',flush=True)
        del model,_tmp; torch.cuda.empty_cache(); gc.collect()
    write_csv(out_dir/'y04_all_frame_metrics.csv',all_rows)
    if a.smoke_only:
        write_json(marker,{'status':'SMOKE_PASS','training_performed':True,'formal_checkpoints_saved':False,'note':'Smoke metrics are mechanics-only.'}); return 0
    valid, seq_summary=metric_summary(all_rows,split)
    write_csv(out_dir/'y04_temporal_tail_metrics.csv',valid); write_csv(out_dir/'y04_sequence_summary.csv',seq_summary); write_csv(out_dir/'y04_runtime_costs.csv',run_meta)
    summary={'status':'MECHANICS_PASS','experiment':'Y04 fair multi-scene adaptation pilot','training_performed':True,
      'formal_steps_per_arm':FORMAL_STEPS,'training_arms':['P_single','O_single','P_multi','O_multi'],
      'evaluation_arms':[x[0] for x in eval_arms],'sequences':list(SEQUENCES),'temporal_tail_start':EVAL_START,
      'single_scene_gt_frames_across_cycle_slots':actual_single_gt,'multi_scene_gt_frames_across_cycle_slots':actual_multi_gt,
      'checkpoint_selection':'final step only; no validation-based selection',
      'interpretation_limit':'Development temporal tails only; no independent blind domain; inspect sequence summary and GT exposure before claiming multi-scene benefit.',
      'counts':{'all_frame_rows':len(all_rows),'scored_temporal_tail_rows':len(valid),'sequence_summary_rows':len(seq_summary)},
      'artifacts':['y04_preflight.json','y04_split_manifest.csv','y04_budget_manifest.json','y04_training_cycle_manifest.json','y04_training_compute.json','y04_all_frame_metrics.csv','y04_temporal_tail_metrics.csv','y04_sequence_summary.csv','y04_runtime_costs.csv']}
    write_json(marker,summary); print(f'[MECHANICS_PASS] Y04 review package data written: {out_dir}',flush=True); return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(f'[Y04_FAIL] {type(exc).__name__}: {exc}',file=sys.stderr,flush=True); raise
