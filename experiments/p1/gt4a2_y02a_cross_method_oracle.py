#!/usr/bin/env python3
"""Y02-A: matched-pixel cross-method residual attribution for SVR-Depth.

Frozen A, Q0-3 trained B3, Q0-3 S2 gain0.5 (C/D), Y02 one-shot
F-only C/D. Per-target GT oracle affine is OFFLINE DIAGNOSTICS ONLY.
The oracle NEVER enters the propagated state; no model training.

Run from repo root; keep audited q03, y02, y01 scripts under experiments/p1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'p1'))

import gt4a2_exp1_train_eval as exp1
import gt2_fixed_clip_repair_train as gt2
import gt4a2_y02_full_cycle_single_intervention as y02
import gt4a2_q0_3_necessity_controls as q03
import gt4a2_y01_natural_error_diagnostics as y01

REFRESH = 32
AGES = (8, 16, 24)
CYCLES = {
    # Locked before executing Y02-A; selection does not use Y02 success/failure labels.
    'crowd2': (2, 10, 20, 24, 26),  # last two are Crowd2 temporal holdout
    'crowd3': (1, 6, 12, 19, 25),
    'balloon2': (1, 4, 7, 10, 13),
}
CKPT_SHA = {
    'C': 'b1cd8f88a5bd45d75af1a6e3050ee1b5188fb545291b5c62ec304038839ae8d3',
    'D': '3ec06cd9ed9128f8a21afa6321312c3449414326a4e50580b939128bcba84f98',
}
B3_SHA = '42b71562dc86ff94b3772cf604dae7a9151c7806c4952275f963fb40ba1eaca7'
MODEL_SHA = '615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e'
GT_LABELS = ('A', 'B3', 'S2_C', 'S2_D', 'F_C', 'F_D')
REFERENCE_MAP = {
    'A': 'A_native',
    'B3': 'B3_output_adapter_trained_output_only',
    'S2_C': 'S2_C_gain0.5_R1_F_only',
    'S2_D': 'S2_D_gain0.5_R1_F_only',
}


def digest(p: Path) -> str:
    sha = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(p: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f'Refusing to create empty results: {p}')
    keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def read_reference(path: Path, member: str, fallback: str) -> tuple[dict, dict]:
    """Return keyed full-frame reference and provenance; no predictions are inferred."""
    if path.is_file():
        with zipfile.ZipFile(path) as zf:
            if member not in zf.namelist():
                raise FileNotFoundError(f'{member} missing in {path}')
            import io
            rows = list(csv.DictReader(io.StringIO(zf.read(member).decode('utf-8-sig'))))
        source = str(path)
        file_sha = digest(path)
    else:
        p = Path(fallback).resolve()
        if not p.is_file():
            raise FileNotFoundError(f'Required reference missing: {path} or {p}')
        with p.open('r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        source = str(p)
        file_sha = digest(p)
    result = {}
    if member.startswith('q0_3'):
        for r in rows:
            key = (r['sequence'], r['arm'], int(r['frame_index_zero_based']))
            if key in result:
                raise RuntimeError(f'Duplicate Q0-3 reference key: {key}')
            result[key] = r
    else:
        for r in rows:
            if str(r.get('refresh_frame', '')).lower() == 'true':
                continue
            key = (r['sequence'], r['checkpoint'], int(r['snapshot_cycle']),
                   int(r['snapshot_age']), r['branch'], int(r['target_index']))
            if key in result:
                raise RuntimeError(f'Duplicate Y02 reference key: {key}')
            result[key] = r
    return result, {'path': source, 'sha256': file_sha, 'rows': len(rows)}


def args_parser():
    p = argparse.ArgumentParser(description='Y02-A matched-frame own-oracle residual attribution; no training.')
    p.add_argument('--snapshot', default=os.environ.get('VDEEP_SNAPSHOT', ''))
    p.add_argument('--sequences', default='crowd2,crowd3,balloon2')
    p.add_argument('--crowd2-pair-csv', default='outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv')
    p.add_argument('--crowd3-pair-csv', default='outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv')
    p.add_argument('--balloon2-pair-csv', default='outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv')
    p.add_argument('--c-checkpoint', default='outputs/gt4a2_c3/gt4a2_c3_C_final_step1000.pt')
    p.add_argument('--d-checkpoint', default='outputs/gt4a2_c3/gt4a2_c3_D_final_step1000.pt')
    p.add_argument('--b3-checkpoint', default='outputs/gt4a2_q0_3_validation/training/Q03_B3_output_adapter_final_step1000.pt')
    p.add_argument('--q03-review', default='outputs/gt4a2_q0_3_validation/q0_3_review.zip')
    p.add_argument('--y02-review', default='outputs/gt4a2_y02_validation/y02_review.zip')
    p.add_argument('--max-cycles-per-sequence', type=int, default=0,
                   help='0 uses locked cycle bank; 1 is a short smoke, never changes sequence metrics')
    p.add_argument('--checkpoint', choices=('C','D','both'), default='both')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--smoke-only', action='store_true')
    p.add_argument('--out-dir', default='outputs/gt4a2_y02a_validation')
    a = p.parse_args()
    a.seqs = [s.strip().lower() for s in a.sequences.split(',') if s.strip()]
    if not a.seqs or len(set(a.seqs)) != len(a.seqs) or set(a.seqs) - set(CYCLES):
        p.error('Bad --sequences; choose unique crowd2,crowd3,balloon2')
    if a.max_cycles_per_sequence < 0 or a.max_cycles_per_sequence > 5:
        p.error('--max-cycles-per-sequence must be in 0..5')
    if a.smoke_only and a.max_cycles_per_sequence == 0:
        a.max_cycles_per_sequence = 1
    a.ckpts = ('C','D') if a.checkpoint == 'both' else (a.checkpoint,)
    return a


def selected_bank(seq: str, limit: int):
    return list(CYCLES[seq][:limit] if limit else CYCLES[seq])


def make_windows(seq: str, n_frames: int, limit: int):
    result = []
    for cycle in selected_bank(seq, limit):
        refresh = (cycle + 1) * REFRESH
        if refresh >= n_frames:
            raise RuntimeError(f'{seq}: cycle {cycle} has no observed refresh frame')
        for age in AGES:
            t = cycle * REFRESH + age
            result.append({'cycle':cycle,'age':age,'index':t,'refresh':refresh,'anchor_index':t+1})
    return result


def preflight(a):
    if not a.snapshot:
        raise RuntimeError('VDEEP_SNAPSHOT is empty; pass --snapshot')
    snapshot = Path(a.snapshot).resolve()
    for name in ('config.json','model.safetensors'):
        if not (snapshot/name).is_file():
            raise FileNotFoundError(snapshot/name)
    if digest(snapshot/'model.safetensors') != MODEL_SHA:
        raise RuntimeError('Base model checkpoint hash differs from Q0-3/Y02')
    source = y02.verify_source_hashes()
    cpaths = {'C':Path(a.c_checkpoint).resolve(),'D':Path(a.d_checkpoint).resolve()}
    for label, path in cpaths.items():
        if not path.is_file() or digest(path) != CKPT_SHA[label]:
            raise RuntimeError(f'{label} C3 final checkpoint missing or SHA mismatch: {path}')
        y02.load_checkpoint_metadata(path, label)
    b3path = Path(a.b3_checkpoint).resolve()
    if not b3path.is_file() or digest(b3path) != B3_SHA:
        raise RuntimeError(f'Q0-3 fair B3 checkpoint missing or SHA mismatch: {b3path}')
    obj = torch.load(b3path,map_location='cpu',weights_only=False)
    if (obj.get('mode') != 'Q03_B3_output_adapter' or int(obj.get('step_count',-1)) != 1000 or
            obj.get('sample_order_sha256') != 'ac251dbb013ceef5958a47a5f863369cc1f317070f4dbd1ba396ad889ab3aad6' or
            obj.get('pretrained_weights_sha256') != MODEL_SHA or
            obj.get('training_rgb_indices_inclusive') != [0,767]):
        raise RuntimeError('Q0-3 B3 training provenance mismatch')
    for l in ('C','D'):
        ck = torch.load(cpaths[l],map_location='cpu',weights_only=False)
        if set(ck['repair_head_state_dict']) != set(obj['adapter_state_dict']):
            raise RuntimeError(f'B3 and C3-{l} adapter head architectures differ')
    pair_paths = {
      'crowd2':Path(a.crowd2_pair_csv).resolve(),
      'crowd3':Path(a.crowd3_pair_csv).resolve(),
      'balloon2':Path(a.balloon2_pair_csv).resolve(),
    }
    manifests={}; rows={}; windows={}
    for seq in a.seqs:
        if not pair_paths[seq].is_file(): raise FileNotFoundError(pair_paths[seq])
        rs=exp1.read_manifest(pair_paths[seq],training=False)
        ws=make_windows(seq,len(rs),a.max_cycles_per_sequence)
        rows[seq]=rs; windows[seq]=ws
        manifests[seq]={'path':str(pair_paths[seq]),'sha256':digest(pair_paths[seq]),
                        'n_frames':len(rs),'cycles':selected_bank(seq,a.max_cycles_per_sequence),
                        'windows':len(ws)}
    qref,qprov=read_reference(Path(a.q03_review),'q0_3_frame_metrics.csv',
                              'outputs/gt4a2_q0_3_validation/q0_3_frame_metrics.csv')
    yref,yprov=read_reference(Path(a.y02_review),'y02_future_frame_metrics.csv',
                              'outputs/gt4a2_y02_validation/y02_future_frame_metrics.csv')
    for seq,ws in windows.items():
        for w in ws:
            for idx in range(w['index']+1, w['refresh']):
                if str(rows[seq][idx].get('gt_match_status','')).upper() != 'MATCHED': continue
                for arm in ('A','B3')+tuple('S2_'+ck for ck in a.ckpts):
                    if (seq,REFERENCE_MAP[arm],idx) not in qref:
                        raise RuntimeError(f'Missing Q03 reference: {(seq,arm,idx)}')
                for ck in a.ckpts:
                    for branch in ('A','F_only'):
                        key=(seq,ck,w['cycle'],w['age'],branch,idx)
                        if key not in yref:
                            raise RuntimeError(f'Missing Y02 reference: {key}')
    report={
        'status':'PREFLIGHT_PASS','experiment':'Y02-A matched-pixel cross-method own-oracle diagnostics',
        'base_model_sha256':MODEL_SHA,'source_checks':source,
        'q0_3_reference':qprov,'y02_reference':yprov,
        'b3_checkpoint':{'path':str(b3path),'sha256':B3_SHA},
        'c3_checkpoints':{k:{'path':str(v),'sha256':CKPT_SHA[k]} for k,v in cpaths.items()},
        'manifests':manifests,
        'protocol':{'K':REFRESH,'ages':AGES,'cycle_bank':CYCLES,
            'selection':'prelocked cycle ids; no Y02 outcome-based window selection',
            'A':'native A continuous',
            'B3':'Q03 final trained output-only full-sequence native recurrent state; no writeback',
            'S2':'Q03 every-propagation-frame post-readout F-only, gain=0.5 from sequence start',
            'F':'Y02 once at age 8/16/24 from natural A state; no subsequent repairs',
            'oracle':'per-target per-method GT OLS; OFFLINE ONLY; not deployment',
            'mask':'intersection of GT validity and all compared branches on same target frame',
            'matched_metric':'raw and own-oracle AbsRel, RMSE, delta1 and signed bias',
            'training':'none','test_claim':'development diagnostics; no held-out domain inference',
        },'script_sha256':digest(Path(__file__).resolve())}
    return report,rows,windows,qref,yref,b3path,cpaths


def calc_metrics(gt,pred,mask):
    return y01.calculate_masked_metrics(gt,pred,mask)


def own_oracle(gt,pred,mask):
    fit=y01.fit_frame_scale_bias(gt,pred,mask)
    scale=float(fit['oracle_scale_gt_from_pred']); bias=float(fit['oracle_bias_m'])
    # Match Y01: clip at 1e-3 to keep physically valid reported depth.
    cal=np.maximum(scale*pred.astype(np.float64)+bias,1e-3).astype(np.float32)
    return fit,calc_metrics(gt,cal,mask),cal


def continuous_predictions(seq, rows, model, shape, device, index_set, arm, head=None):
    """Run from first RGB frame, preserving the exact q03 output/state semantics."""
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    saved={}; prev=None; radius=confidence=features=rays_map=None
    for i,rec in enumerate(rows):
        matched=str(rec.get('gt_match_status','')).upper()=='MATCHED'
        frame=exp1.load_frame(rec,shape,device,need_gt=(matched and i in index_set))
        with torch.no_grad():
            if i % REFRESH == 0:
                rays_map,radius,confidence,features=exp1.keyframe_state(model,frame,shape)
                out_r=radius
            else:
                static=exp1.transition_static(model,prev,frame,shape,dis)
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
                    radius,confidence,features=exp1.one_fast_step(
                        model,prev,frame,static,radius,confidence,features)
                    out_r=radius
                    if arm == 'B3':
                        adapted=head(features)
                        out_r=q03.radius_from_features(model,adapted)
                        del adapted
                    elif arm.startswith('S2_'):
                        features=q03.apply_repair_residual(features,head,0.5)
                del static
            if i in index_set:
                pred=y02.pred_numpy(out_r,rays_map,shape,model)
                saved[i]=pred
        prev=frame
        if i == 0 or (i+1)%250 == 0 or i==len(rows)-1:
            print(f'[{seq}/{arm}] {i+1}/{len(rows)}',flush=True)
    if saved.keys()!=index_set:
        raise RuntimeError(f'{seq}/{arm}: saved {len(saved)} != expected {len(index_set)}')
    return saved


def expected_metric_check(seq,arm,idx,gt,pred,ref,tolerance=2e-5):
    row=ref[(seq,REFERENCE_MAP[arm],idx)]
    m=gt2.calculate_metrics(gt,pred)
    devs={k:abs(float(row[k])-float(m[k])) for k in ('AbsRel','RMSE','delta1')}
    if max(devs.values())>tolerance:
        raise RuntimeError(f'Q0-3 replay mismatch {seq}/{arm}/frame{idx}: {devs}. '
                           'Stop; do not interpret oracle results.')
    return max(devs.values())


def y02_metric_check(seq,ck,w,idx,gt,pred,yref,tolerance=2e-5):
    row=yref[(seq,ck,w['cycle'],w['age'],'F_only',idx)]
    m=gt2.calculate_metrics(gt,pred)
    dev=max(abs(float(row[k])-float(m[k])) for k in ('AbsRel','RMSE','delta1'))
    if dev>tolerance:
        raise RuntimeError(f'Y02 replay mismatch {seq}/{ck}/cycle{w["cycle"]}/age{w["age"]}/frame{idx}: {dev}')
    return dev


def group_summary(rows):
    grouped=defaultdict(list)
    for r in rows:
        k=(r['sequence'],r['split'],r['checkpoint'],r['arm'],r['snapshot_age'])
        grouped[k].append(r)
    out=[]
    for k,rr in sorted(grouped.items()):
        rr=sorted(rr,key=lambda x:(x['cycle'],x['target_index']))
        base={x:(np.asarray([float(r[x]) for r in rr])) for x in (
            'raw_AbsRel','oracle_AbsRel','raw_RMSE','oracle_RMSE','raw_delta1',
            'oracle_delta1','raw_signed_bias_mean_m','oracle_scale')}
        out.append(dict(zip(('sequence','split','checkpoint','arm','snapshot_age'),k))|
                   {'n_paired_window_frame_records':len(rr),
                    **{f'mean_{f}':float(v.mean()) for f,v in base.items()},
                    'fraction_oracle_RMSE_reduction_from_raw':float(
                        1-base['oracle_RMSE'].mean()/max(base['raw_RMSE'].mean(),1e-10)),
                    'note':'Overlapping targets and within-sequence windows are correlated.'})
    return out


def delta_summary(rows):
    g=defaultdict(dict)
    for r in rows:
        key=(r['sequence'],r['checkpoint'],r['cycle'],r['snapshot_age'],r['target_index'])
        g[key][r['arm']]=r
    out=[]
    for key,arms in sorted(g.items()):
        if set(arms)!=set(['A','B3','S2','F']):
            raise RuntimeError(f'Missing matched arm in {key}: {sorted(arms)}')
        a=arms['A']
        for branch in ('B3','S2','F'):
            b=arms[branch]
            out.append({'sequence':key[0],'checkpoint':key[1],
                        'cycle':key[2],'snapshot_age':key[3],'target_index':key[4],
                        'split':a['split'],'branch':branch,'n_pixels':a['valid_pixels'],
                        'raw_gain_AbsRel':a['raw_AbsRel']-b['raw_AbsRel'],
                        'oracle_gain_AbsRel':a['oracle_AbsRel']-b['oracle_AbsRel'],
                        'raw_gain_RMSE':a['raw_RMSE']-b['raw_RMSE'],
                        'oracle_gain_RMSE':a['oracle_RMSE']-b['oracle_RMSE'],
                        'raw_harm':int(b['raw_AbsRel']>a['raw_AbsRel']),
                        'oracle_harm':int(b['oracle_AbsRel']>a['oracle_AbsRel']),
                        'future_horizon':b['horizon'],
                        'raw_signed_bias_delta_m':b['raw_signed_bias_mean_m']-a['raw_signed_bias_mean_m'],
                        'oracle_scale_delta':b['oracle_scale']-a['oracle_scale']})
        # Pair F and B3 using the same mask, target and age, with separate oracle fits.
        b=arms['B3']; f=arms['F']
        out.append({'sequence':key[0],'checkpoint':key[1],
                    'cycle':key[2],'snapshot_age':key[3],'target_index':key[4],
                    'split':a['split'],'branch':'F_minus_B3','n_pixels':a['valid_pixels'],
                    'raw_gain_AbsRel':b['raw_AbsRel']-f['raw_AbsRel'],
                    'oracle_gain_AbsRel':b['oracle_AbsRel']-f['oracle_AbsRel'],
                    'raw_gain_RMSE':b['raw_RMSE']-f['raw_RMSE'],
                    'oracle_gain_RMSE':b['oracle_RMSE']-f['oracle_RMSE'],
                    'raw_harm':int(f['raw_AbsRel']>b['raw_AbsRel']),
                    'oracle_harm':int(f['oracle_AbsRel']>b['oracle_AbsRel']),
                    'future_horizon':f['horizon'],
                    'raw_signed_bias_delta_m':f['raw_signed_bias_mean_m']-b['raw_signed_bias_mean_m'],
                    'oracle_scale_delta':f['oracle_scale']-b['oracle_scale']})
        # Additional paired comparators help separate persistent frequency from output-only.
        for other, name in (('S2','F_minus_S2'),('B3','S2_minus_B3')):
            left=arms['F'] if name=='F_minus_S2' else arms['S2']
            right=arms[other]
            out.append({'sequence':key[0],'checkpoint':key[1],
                        'cycle':key[2],'snapshot_age':key[3],'target_index':key[4],
                        'split':a['split'],'branch':name,'n_pixels':a['valid_pixels'],
                        'raw_gain_AbsRel':right['raw_AbsRel']-left['raw_AbsRel'],
                        'oracle_gain_AbsRel':right['oracle_AbsRel']-left['oracle_AbsRel'],
                        'raw_gain_RMSE':right['raw_RMSE']-left['raw_RMSE'],
                        'oracle_gain_RMSE':right['oracle_RMSE']-left['oracle_RMSE'],
                        'raw_harm':int(left['raw_AbsRel']>right['raw_AbsRel']),
                        'oracle_harm':int(left['oracle_AbsRel']>right['oracle_AbsRel']),
                        'future_horizon':left['horizon'],
                        'raw_signed_bias_delta_m':left['raw_signed_bias_mean_m']-right['raw_signed_bias_mean_m'],
                        'oracle_scale_delta':left['oracle_scale']-right['oracle_scale']})
    return out


def window_summary(delta_rows):
    bywin=defaultdict(list)
    for r in delta_rows:
        k=(r['sequence'],r['checkpoint'],r['cycle'],r['snapshot_age'],r['branch'])
        bywin[k].append(r)
    out=[]
    for k,rr in sorted(bywin.items()):
        out.append(dict(zip(('sequence','checkpoint','cycle','snapshot_age','branch'),k))|
                   {'n_future_gt_frames':len(rr),
                    'raw_mean_gain_AbsRel':float(np.mean([r['raw_gain_AbsRel'] for r in rr])),
                    'raw_cumulative_gain_AbsRel':float(sum(r['raw_gain_AbsRel'] for r in rr)),
                    'oracle_mean_gain_AbsRel':float(np.mean([r['oracle_gain_AbsRel'] for r in rr])),
                    'oracle_mean_gain_RMSE':float(np.mean([r['oracle_gain_RMSE'] for r in rr])),
                    'oracle_cumulative_gain_RMSE':float(sum(r['oracle_gain_RMSE'] for r in rr)),
                    'raw_harmed_frame_fraction':float(np.mean([r['raw_harm'] for r in rr])),
                    'oracle_harmed_frame_fraction':float(np.mean([r['oracle_harm'] for r in rr])),
                    'worst_oracle_gain_RMSE':float(min(r['oracle_gain_RMSE'] for r in rr))})
    return out


def review_archive(out_dir,source):
    names=('y02a_preflight.json','y02a_frame_oracles.csv','y02a_paired_gains.csv',
           'y02a_window_summary.csv','y02a_group_summary.csv',
           'y02a_reference_invariants.csv','y02a_summary.json','Y02A_README.md')
    dest=out_dir/'y02a_review.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as zf:
        for n in names:
            zf.write(out_dir/n,n)
        zf.write(source,'gt4a2_y02a_cross_method_oracle.py')
        for f in sorted((out_dir/'selected_examples').glob('*.npz')):
            zf.write(f,f'selected_examples/{f.name}')
    return dest


def main():
    a=args_parser(); out_dir=Path(a.out_dir).resolve(); out_dir.mkdir(parents=True,exist_ok=True)
    pre,manifests,windows,qref,yref,b3path,cpaths=preflight(a)
    write_json(out_dir/'y02a_preflight.json',pre)
    print(f"[PASS] Y02-A preflight: sequences={a.seqs}, windows={sum(len(x) for x in windows.values())}, C/D={a.ckpts}",flush=True)
    if a.preflight_only:
        print('Y02-A preflight complete; no training or inference.',flush=True);return 0
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for frozen VeloDepth replay')
    device=torch.device('cuda');torch.set_num_threads(4)
    torch.manual_seed(20261025);np.random.seed(20261025)
    t0=time.perf_counter();rows_out=[]; invariant=[]
    examples=out_dir/'selected_examples';examples.mkdir(exist_ok=True)
    max_q=max_y=max_native=0.0; expected=0
    for seq in a.seqs:
        rows=manifests[seq];ws=windows[seq]
        targets={i for w in ws for i in range(w['index']+1,w['refresh'])
                 if str(rows[i].get('gt_match_status','')).upper()=='MATCHED'}
        model=gt2.load_model(Path(a.snapshot).resolve(),'off',device)
        shape=exp1.shape_info_for(model,Path(rows[0]['rgb_path']),device)
        print(f'[{seq}] frozen continuous baselines; target GT frames={len(targets)}',flush=True)
        continuous={'A': continuous_predictions(seq,rows,model,shape,device,targets,'A')}
        b3obj=torch.load(b3path,map_location='cpu',weights_only=False)
        b3=q03.make_frozen_head(b3obj['adapter_state_dict'],device)
        continuous['B3']=continuous_predictions(seq,rows,model,shape,device,targets,'B3',b3)
        del b3,b3obj
        heads={}
        for ck in a.ckpts:
            obj=torch.load(cpaths[ck],map_location='cpu',weights_only=False)
            heads[ck]=q03.make_frozen_head(obj['repair_head_state_dict'],device)
            continuous['S2_'+ck]=continuous_predictions(
                seq,rows,model,shape,device,targets,'S2_'+ck,heads[ck])
            del obj
        # Natural A snapshots reused across age- and checkpoint-matched interventions.
        cache=exp1.precompute_a_states(model,rows,
            [{'anchor_index':w['anchor_index'],'refresh_cycle':w['cycle']} for w in ws],
            shape,device)
        for w in ws:
            st=y02.state_to_device(cache[w['anchor_index']],device)
            frames=y02.make_window_frames(rows,w['index'],w['refresh'],shape,device)
            static=y02.precompute_window_static(model,frames,w['index'],w['refresh'],shape)
            pred_a_rows,pred_a=y02.replay_future(
                model,frames,static,shape,w['index'],w['refresh'],st,
                seq,'A',w['cycle'],w['age'],'A')
            for ck in a.ckpts:
                with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
                    fprime=[z.detach() for z in heads[ck](st['features'])]
                fst={'radius':st['radius'],'confidence':st['confidence'],
                     'features':fprime,'rays_map':st['rays_map']}
                _,pred_f=y02.replay_future(
                    model,frames,static,shape,w['index'],w['refresh'],fst,
                    seq,ck,w['cycle'],w['age'],'F_only')
                for h in range(1,w['refresh']-w['index']):
                    idx=w['index']+h
                    if idx not in targets:continue
                    gt=frames[idx]['gt_np']
                    predicted={'A':pred_a[h],'B3':continuous['B3'][idx],
                               'S2':continuous['S2_'+ck][idx],'F':pred_f[h]}
                    diff=float(np.max(np.abs(pred_a[h].astype(np.float32)
                                              -continuous['A'][idx].astype(np.float32))))
                    max_native=max(max_native,diff)
                    if diff>1e-3:
                        raise RuntimeError(f'A trajectory mismatch {seq} cycle{w["cycle"]} idx{idx}: {diff}')
                    mask=(gt>0)&np.isfinite(gt)
                    for p in predicted.values():
                        mask &= np.isfinite(p)&(p>0)
                    nmask=int(mask.sum())
                    if nmask<1000:
                        raise RuntimeError(f'Too few common GT pixels at {seq}/{idx}: {nmask}')
                    full_mask_n=int(((gt>0)&np.isfinite(gt)).sum())
                    if nmask != full_mask_n:
                        raise RuntimeError(
                            f'Branch-specific invalid predictions alter common mask at {seq}/{idx}: '
                            f'{nmask} common vs {full_mask_n} GT-valid; stop and diagnose.'
                        )
                    # Independently verify original Q03/Y02 metrics on original per-arm mask.
                    for arm in ('A','B3','S2'):
                        actual='S2_'+ck if arm=='S2' else arm
                        d=expected_metric_check(seq,actual,idx,gt,predicted[arm],qref)
                        max_q=max(max_q,d)
                    dy=y02_metric_check(seq,ck,w,idx,gt,predicted['F'],yref)
                    max_y=max(max_y,dy)
                    for arm,pred in predicted.items():
                        raw=calc_metrics(gt,pred,mask)
                        fit,oracle,cal=own_oracle(gt,pred,mask)
                        oracle_rmse=float(oracle['RMSE'])
                        # Unclipped GT-OLS should not make per-target RMSE worse
                        # (clipping may change it; check with tolerance only).
                        if oracle_rmse > float(raw['RMSE']) + 2e-3:
                            raise RuntimeError(f'Unexpected large oracle RMSE degradation {seq}/{idx}/{arm}')
                        rows_out.append({
                            'sequence':seq,
                            'split': 'crowd2_temporal_holdout' if seq=='crowd2' and idx>=768
                                     else 'crowd2_train_range' if seq=='crowd2'
                                     else 'cross_sequence_development',
                            'checkpoint':ck,'cycle':w['cycle'],'snapshot_age':w['age'],
                            'snapshot_index':w['index'],'target_index':idx,'horizon':h,
                            'arm':arm,'valid_pixels':nmask,
                            'gt_valid_pixels':full_mask_n,
                            'rgb_file':Path(rows[idx]['rgb_path']).name,
                            'gt_file':Path(rows[idx]['depth_path']).name,
                            **{f'raw_{k}':v for k,v in raw.items() if k!='n_pixels'},
                            **{f'oracle_{k}':v for k,v in oracle.items() if k!='n_pixels'},
                            'oracle_scale':fit['oracle_scale_gt_from_pred'],
                            'oracle_bias_m':fit['oracle_bias_m'],
                            'oracle_fit_r2':fit['oracle_fit_r2'],
                            'oracle_nonpositive_fraction_before_clamp':float(
                                np.mean((float(fit['oracle_scale_gt_from_pred'])*pred.astype(np.float64)+
                                float(fit['oracle_bias_m']))[mask]<=0)),
                        })
                    invariant.append({'sequence':seq,'checkpoint':ck,
                                      'cycle':w['cycle'],'snapshot_age':w['age'],
                                      'target_index':idx,'A_replay_max_abs_m':diff,
                                      'Q03_metric_max_deviation':max_q,
                                      'Y02_F_metric_max_deviation':dy,
                                      'same_gt_mask_pixels':nmask,
                                      'gt_valid_pixels':full_mask_n})
                    # Archive a small set of exact selected predictions and GT/mask.
                    if w['cycle']==selected_bank(seq,a.max_cycles_per_sequence)[0] and w['age']==16 and h in (1,7):
                        fpath=examples/f'{seq}_{ck}_cycle{w["cycle"]}_age16_h{h}.npz'
                        np.savez_compressed(fpath,
                           gt=gt.astype(np.float32),mask=mask.astype(np.uint8),
                           A=predicted['A'].astype(np.float32),
                           B3=predicted['B3'].astype(np.float32),
                           S2=predicted['S2'].astype(np.float32),
                           F=predicted['F'].astype(np.float32))
                    expected+=1
                # Next scheduled refresh keyframe must erase the F-only intervention.
                refresh_h=w['refresh']-w['index']
                reset_diff=float(np.max(np.abs(pred_a[refresh_h].astype(np.float32)
                                                -pred_f[refresh_h].astype(np.float32))))
                if reset_diff>1e-3:
                    raise RuntimeError(f'F-only did not reset at keyframe: {seq}/{w["cycle"]}/{ck}')
                del fst,fprime,pred_f
            del frames,static,st,pred_a,pred_a_rows
            print(f'[{seq}] cycle={w["cycle"]} age={w["age"]} C/D replay complete',flush=True)
        del heads,cache,model,continuous
        torch.cuda.empty_cache()
    paired=delta_summary(rows_out)
    win=window_summary(paired)
    group=group_summary(rows_out)
    write_csv(out_dir/'y02a_frame_oracles.csv',rows_out)
    write_csv(out_dir/'y02a_paired_gains.csv',paired)
    write_csv(out_dir/'y02a_window_summary.csv',win)
    write_csv(out_dir/'y02a_group_summary.csv',group)
    write_csv(out_dir/'y02a_reference_invariants.csv',invariant)
    summary={'status':'MECHANICS_PASS','experiment':'Y02-A cross-method own-oracle residual attribution',
             'training_performed':False,'partial_smoke':bool(a.smoke_only or a.max_cycles_per_sequence),
             'sequences':a.seqs,'checkpoints':a.ckpts,'K':REFRESH,'snapshot_ages':AGES,
             'preselected_cycles':{s:selected_bank(s,a.max_cycles_per_sequence) for s in a.seqs},
             'counts':{'matched_snapshot_target_comparisons_across_checkpoints':expected,
                       'frame_oracle_rows':len(rows_out),'paired_gain_rows':len(paired),
                       'window_summary_rows':len(win),'reference_invariant_rows':len(invariant)},
             'mechanics':{'A_native_replay_max_abs_m':max_native,
                          'Q03_reference_max_metric_deviation':max_q,
                          'Y02_F_reference_max_metric_deviation':max_y,
                          'same_mask_verified':True,'refresh_reset_verified':True},
             'interpretation_limits':['Per-method oracle uses target GT and is offline only.',
                'Oracle RMSE directly targets least-squares; oracle AbsRel need not always fall.',
                'Selected windows were fixed before evaluation; cannot claim independent domain.',
                'Each target can appear in multiple overlapping windows; report cluster/group summaries.',
                'B3 and S2 are full-sequence from frame0; F is single-write replay from native A.',
                'Equal output masks are enforced; no oracle outputs are fed to future state.'],
             'runtime_sec':time.perf_counter()-t0,
             'run_script_sha256':digest(Path(__file__).resolve()),
             'preflight_sha256':digest(out_dir/'y02a_preflight.json')}
    write_json(out_dir/'y02a_summary.json',summary)
    (out_dir/'Y02A_README.md').write_text(
        '# Y02-A Cross-Method Residual Attribution\n\n'
        'All arms are scored on identical GT frame/mask. Each arm fits its own per-frame GT-based '
        'scale/bias oracle after inference, never inside the online state. '\
        'Gain positive means the comparator has lower error; '
        'F_minus_B3 positive means F is better than B3.\n'
        'Preselected cycle bank is locked in script. Overlapping windows are correlated.\n',encoding='utf-8')
    z=review_archive(out_dir,Path(__file__).resolve())
    print(f'[MECHANICS_PASS] Y02-A: {z}; rows={len(rows_out)}, runtime={summary["runtime_sec"]:.1f}s',flush=True)
    return 0

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise
