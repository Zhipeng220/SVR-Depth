#!/usr/bin/env python3
"""Offline protocol/contract tests; real Torch/CUDA remains a server-side canary."""
from pathlib import Path
import hashlib, json, subprocess, sys, types
import y05r_common as c

ROOT = Path(__file__).resolve().parent


def check(ok, msg):
    if not ok: raise AssertionError(msg)


def main():
    s = c.frozen_settings(False)
    check(c.ARMS == ('C_star','CF'), 'Y05-R must have exactly two arms')
    check(s['fixed_updates'] == 1000 and s['horizon'] == 4 and s['train_K'] == 32, 'Formal settings drift')
    check(s['objective_C_star'].startswith('current_AbsRel'), 'C* objective must be current-only')
    check(s['objective_CF'].startswith('current_AbsRel + mean(available H1..H4'), 'CF must retain current target and add H4')
    check(s['future_weight'] == 1.0 and s['harm_penalty'] is False and s['direction_gate'] is False, 'Unapproved method branch added')
    check(c.gate_tolerance([]) == 1e-4, 'Frozen minimum practical margin drift')
    check(abs(c.gate_tolerance([1e-5]) - 1e-4) < 1e-12, 'Repeat margin floor drift')
    check(abs(c.gate_tolerance([1e-3]) - 3e-3) < 1e-12, 'Repeat margin multiplier drift')
    check(c.canonical({'b':2,'a':1}) == c.canonical({'a':1,'b':2}), 'Canonical protocol hash unstable')
    for name in ('gt4a2_y05r.py','y05r_train.py','y05r_eval.py'):
        text=(ROOT/name).read_text()
        compile(text, str(ROOT/name), 'exec')
    # Import the aggregation layer with a stub engine to test paired H4 arithmetic offline.
    stub=types.ModuleType('y05_engine')
    for name in ('Engine','ensure_bank','parent_for','detached_state','numpy_metrics','tsave','depth_loss','valid_state'):
        setattr(stub,name,lambda *a,**k: None)
    sys.modules['y05_engine']=stub
    import y05r_eval
    import y05r_train
    frames=[]
    for arm,repair_abs in (('C_star',0.09),('CF',0.08)):
        for action,absrel in (('skip',0.10),('repair',repair_abs)):
            for h in range(5):
                frames.append(dict(parent_source='common_C_S8',arm=arm,action=action,case_id='case1',sequence='balloon2',anchor=8,h=h,matched=True,
                    AbsRel=absrel if h>0 else 0.11, RMSE=0.2 if action=='skip' else 0.18, delta1=0.8 if action=='skip' else 0.85))
    windows, summary = y05r_eval.h4_aggregate(frames)
    check(len(windows)==2, 'H4 aggregation did not produce one paired window per arm')
    values={r['arm']:r['repair_minus_skip_H4_AbsRel'] for r in windows}
    check(abs(values['C_star']+0.01)<1e-12 and abs(values['CF']+0.02)<1e-12, 'H4 paired delta arithmetic incorrect')
    check(all(r['harmed_future_frames_gt0p01']==0 for r in windows), 'H4 harmed-frame threshold arithmetic incorrect')
    # Exercise the natural-summary join using exactly matched GT frames for all three videos.
    class FakeCtx: pass
    ctx=FakeCtx();ctx.refs={};ctx.old_natural={}
    natural=[]; repeat=[]
    for seq in c.SEQS:
        for i in (300,332):
            ctx.refs[('A_K32_g0_f1/raw',seq,i)]={'matched':'true','AbsRel':0.10,'RMSE':0.20,'delta1':0.80,'valid_pixels':10}
            ctx.old_natural[('C_current_K32_S8_g1',seq,i)]={'matched':'true','AbsRel':0.09,'RMSE':0.19,'delta1':0.82,'valid_pixels':10}
            for arm,base in (('C_star',0.085),('CF',0.080)):
                row=dict(arm=arm,sequence=seq,index=i,split='tail',scored=True,AbsRel=base,RMSE=0.18,delta1=0.83,valid_pixels=10,prediction_sha256=f'{arm}-{seq}-{i}')
                natural.append(row);repeat.append(dict(row))
    summaries, repeats, per_seq, tol = y05r_eval.natural_summary(ctx,natural,repeat)
    check(len([r for r in summaries if r['sequence']=='MACRO'])==4, 'Natural macro summary rows missing')
    check(abs(tol-1e-4)<1e-12 and abs(per_seq['CF','crowd2']['AbsRel']-0.08)<1e-12, 'Natural summary/repeat tolerance arithmetic incorrect')
    risk=y05r_eval.natural_risk_rows(ctx,natural)
    check(len(risk)==6 and all(r['severe_harmed_frames_gt0p01']==0 for r in risk), 'Natural severe-harm calculation incorrect')
    # The objective implementation must never replace current supervision with future-only supervision.
    train=(ROOT/'y05r_train.py').read_text()
    check("objective = current + (future_mean if kind == 'CF' else 0.0) + mu * delta" in train, 'Y05-R objective contract changed')
    check("kind in ARMS" in train, 'Unexpected training arm contract')
    check("matched_future_gt = int(result['future_GT'])" in train, 'Gradient audit must capture future GT count before releasing result')
    check("matched_future_GT=matched_future_gt" in train, 'Gradient audit record must use captured future GT count')
    check("del negative, frames, statics, parent" in train, 'Gradient audit must not delete result/g twice')
    check("committed_future_GT_available_for_diagnostics=sum(int(r['future_GT']) for r in curve)" in train,
          'Training summary must use the future_GT key recorded in each update')
    check("committed_future_GT_supervision_exposures=(sum(int(r['future_GT']) for r in curve) if kind == 'CF' else 0)" in train,
          'Future GT supervision exposure accounting key mismatch')
    print('[PASS] Y05-R offline protocol tests; no model/GPU execution was attempted.')

if __name__ == '__main__': main()
