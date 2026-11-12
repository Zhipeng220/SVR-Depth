#!/usr/bin/env python3
"""Freeze M02/M04 plans and aggregate externally generated, fully paired streams."""
import argparse, math, statistics
from collections import defaultdict
from pathlib import Path
from m01 import check, sha, read, rows, write_json, write_csv, percentile

FAMILIES=('A','O_fast','O_multi','P')
ARMS=('OFF','OUTPUT_ONLY','F_ONLY','P_NATIVE')

def verify_parent(parent):
    p=read(parent)
    check(p.get('protocol')=='M01-v1','Requires frozen M01-v1 protocol')
    check(p.get('test_unsealed') is False and p.get('independence_audited') is True and p.get('depth_contract_confirmed') is True,'Invalid M01 audit attestations')
    for path,digest in p['input_sha256'].items(): check(Path(path).is_file() and sha(path)==digest, f'Frozen input changed: {path}')
    return p

def freeze(config_path,parent_path,out):
    c=read(config_path); parent_path=Path(parent_path).resolve(); parent=verify_parent(parent_path)
    check(c.get('experiment') in ('M02','M04'),'Unknown experiment')
    check(c.get('test_unsealed') is False,'Test is unsealed or status is unspecified')
    check(c.get('m00_state_and_depth_audit_complete') is True,'M00 state/depth audit required')
    check(c.get('selection_before_test') is True,'Selection must be frozen before test')
    kind=c['experiment']; units=[]
    if kind=='M02':
        check(c.get('development_access_equal') is True,'Document equal development access and candidate budget')
        evidence=Path(c.get('development_selection_evidence','')).resolve()
        check(evidence.is_file(),'Development selection evidence missing')
        c['development_selection_evidence']=str(evidence); c['development_selection_sha256']=sha(evidence)
        budget=c['candidate_counts']; check(set(budget)==set(FAMILIES),'Declare candidate budget per family')
        check(all(isinstance(v,int) and 1<=v<=12 for v in budget.values()),'Candidate counts must be 1..12')
        check(len(set(budget.values()))==1,'Candidate budgets must be equal')
        points=c['points']; check(len({x['id'] for x in points})==len(points),'Duplicate point ID')
        check(all(x['family'] in FAMILIES and x['k'] in (16,32,64) and x['role'] in ('primary','selected','alternative') and x.get('parameter_semantics') for x in points),'Invalid point')
        for family in FAMILIES:
            for k in (16,32,64):
                matches=[x for x in points if x['family']==family and x['k']==k and x['role'] in ('primary','selected')]
                check(len(matches)==1,f'Need exactly one selected point for {family} K{k}')
        for x in points:
            check(x['role']!='primary' or (x['family']=='P' and x['k']==16),'Only P K16 is primary')
            if x['family']=='P' and x['k']==16:
                check(x['role']=='primary' and x.get('gain')==1 and x.get('frequency')==1 and x.get('alignment')=='raw','P primary must be K16/g1/f1/raw')
            if x['role']=='alternative':
                check(x['family']=='P' and x['k']==32 and x.get('gain')==.5 and x.get('frequency')==1 and x.get('alignment')=='raw','Only P-Alt K32/g0.5/f1/raw is defined')
        check(sum(x['role']=='alternative' for x in points)<=1,'Duplicate P-Alt')
        for x in points:
            seeds=['native'] if x['family']=='A' else [str(z['seed']) for z in parent['checkpoints'][x['family']]]
            for seed in seeds: units.append(dict(id=x['id'],family=x['family'],k=x['k'],role=x['role'],seed=seed))
    else:
        check(c.get('same_head_identity_and_state_semantics_audited') is True,'Audit P head identity and all persistent state fields')
        check(c.get('k')==16 and c.get('gain')==1 and c.get('frequency')==1 and c.get('alignment')=='raw','M04 uses K16/g1/f1/raw')
        for arm in ARMS:
            for cp in parent['checkpoints']['P']:
                units.append(dict(id=arm,family='P_same_head',k=16,role='continuous',seed=str(cp['seed'])))
    c['units']=units; c['m01_protocol']=str(parent_path); c['m01_sha256']=sha(parent_path)
    c['source_sha256']={str(Path(__file__).resolve()):sha(__file__),str(Path(__file__).with_name('m01.py').resolve()):sha(Path(__file__).with_name('m01.py'))}
    c['interpretation']='M02 fixed development-selected points; M04 continuous strategies with divergent histories, not same-parent causal effects.'
    write_json(out,c); print(f'{kind} frozen: {len(units)} units per recording; {out}')

def load_plan(path):
    c=read(path); check(sha(c['m01_protocol'])==c['m01_sha256'],'M01 protocol changed')
    for p,h in c['source_sha256'].items(): check(Path(p).is_file() and sha(p)==h,'Analysis source changed: '+p)
    if c['experiment']=='M02': check(sha(c['development_selection_evidence'])==c['development_selection_sha256'],'Development evidence changed')
    return c,verify_parent(c['m01_protocol'])

def report(plan_path, frame_path, timing_path, out):
    c,parent=load_plan(plan_path); kind=c['experiment']; seqs={s['sequence']:s for s in parent['sequences']}
    units={(u['id'],u['seed']):u for u in c['units']}; check(len(units)==len(c['units']),'Duplicate units')
    data=defaultdict(dict)
    for r in rows(frame_path):
        key=(r['sequence'],r['unit'],r['seed']); check(key[0] in seqs and key[1:] in units,'Unexpected frame unit')
        i=int(r['index']); check(i not in data[key],'Duplicate frame'); data[key][i]=r
    timing=defaultdict(list)
    for r in rows(timing_path):
        key=(r['sequence'],r['unit'],r['seed']); check(key[0] in seqs and key[1:] in units,'Unexpected timing unit')
        check(r['verified']=='true','Unverified timing'); ms=float(r['mean_ms']); check(math.isfinite(ms) and ms>0,'Invalid timing')
        timing[key].append((r['repeat'],ms))
    vids=[]; comparisons=[]
    for seq,meta in seqs.items():
        manifest=rows(meta['manifest']); n=len(manifest)
        for (unit,seed),u in units.items():
            key=(seq,unit,seed); rr=data[key]
            check(set(rr)==set(range(n)),f'Incomplete RGB stream: {key}')
            times=timing[key]; check(len(times)==3 and len({x[0] for x in times})==3, f'Need 3 distinct verified timings: {key}')
            if kind=='M04': ref=('OFF',seed)
            else:
                refpoint=next(x for x in c['points'] if x['family']=='A' and x['k']==u['k'] and x['role'] in ('selected','primary'))
                ref=(refpoint['id'],'native')
            refrows=data[(seq,*ref)]; check(set(refrows)==set(range(n)),f'Incomplete reference stream: {seq}/{ref}')
            dif=[]; metrics=defaultdict(list); failed=0; matched_count=0
            for i,src in enumerate(manifest):
                r=rr[i]; a=refrows[i]; status=r['status']; ast=a['status']; matched=src['gt_match_status'].upper()=='MATCHED'
                check(status in ('ok','unmatched','invalid_gt','invalid_prediction') and ast in ('ok','unmatched','invalid_gt','invalid_prediction'),'Invalid status')
                check((status!='unmatched')==matched and (ast!='unmatched')==matched,'GT matching mismatch')
                check((status=='invalid_gt')==(ast=='invalid_gt'),'GT validity mismatch')
                if not matched: continue
                matched_count+=1
                if status in ('ok','invalid_prediction'):
                    check(r['mask_sha256'] and r['mask_sha256']==a['mask_sha256'] and r['valid_pixels']==a['valid_pixels'] and int(r['valid_pixels'])>0,'GT mask mismatch')
                if status=='invalid_prediction' or ast=='invalid_prediction': failed+=1; continue
                if status=='invalid_gt': continue
                for f in ('AbsRel','RMSE','delta1'):
                    v=float(r[f]); vref=float(a[f]); check(math.isfinite(v) and math.isfinite(vref),'Nonfinite metrics')
                    check((0<=v<=1 and 0<=vref<=1) if f=='delta1' else (v>=0 and vref>=0),'Metric outside range'); metrics[f].append(v)
                dif.append(float(r['AbsRel'])-float(a['AbsRel']))
            check(dif,f'No valid paired frames: {key}')
            vids.append(dict(dataset=meta['dataset'],scene=meta['physical_scene_group'],sequence=seq,unit=unit,family=u['family'],k=u['k'],role=u['role'],seed=seed,reference=ref[0],RGB=n,matched=matched_count,valid_paired=len(dif),failed_paired=failed,
                AbsRel=statistics.mean(metrics['AbsRel']),RMSE=statistics.mean(metrics['RMSE']),delta1=statistics.mean(metrics['delta1']),p95_AbsRel=percentile(metrics['AbsRel'],.95),delta_AbsRel=statistics.mean(dif),harmed_fraction=sum(x>0 for x in dif)/len(dif),severe_fraction=sum(x>.01 for x in dif)/len(dif),latency_ms=statistics.median(x[1] for x in times)))
    fields=('AbsRel','RMSE','delta1','p95_AbsRel','delta_AbsRel','harmed_fraction','severe_fraction','latency_ms')
    grouped=defaultdict(list)
    for r in vids: grouped[(r['dataset'],r['scene'],r['unit'],r['seed'])].append(r)
    scenes=[dict(dataset=d,scene=g,unit=u,seed=s,recordings=len(v),**{k:statistics.mean(x[k] for x in v) for k in fields}) for (d,g,u,s),v in sorted(grouped.items())]
    macros=defaultdict(list)
    for r in scenes: macros[(r['dataset'],r['unit'],r['seed'])].append(r)
    summary=[dict(dataset=d,unit=u,seed=s,scene_groups=len(v),**{k:statistics.mean(x[k] for x in v) for k in fields}) for (d,u,s),v in sorted(macros.items())]
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    check(not any((out/n).exists() for n in ('per_recording.csv','per_scene.csv','scene_macro.csv','summary.json')),'Output already exists')
    write_csv(out/'per_recording.csv',vids); write_csv(out/'per_scene.csv',scenes); write_csv(out/'scene_macro.csv',summary)
    write_json(out/'summary.json',dict(status='PARTIAL_ANALYSIS',experiment=kind,plan_sha256=sha(plan_path),frames_sha256=sha(frame_path),timing_sha256=sha(timing_path),interpretation=c['interpretation'],limitations='Conditional on valid paired frames; failure counts must be reported. No confidence intervals, phase or horizon analysis, p50/p95 latency, VRAM, call accounting, selection verification, or independent-scene proof.'))
    print('Partial analysis:',out)

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd',required=True)
    f=s.add_parser('freeze'); f.add_argument('--config',required=True); f.add_argument('--m01-protocol',required=True); f.add_argument('--out',required=True); f.add_argument('--expected-experiment',choices=('M02','M04'))
    r=s.add_parser('report'); r.add_argument('--plan',required=True); r.add_argument('--frames',required=True); r.add_argument('--timing',required=True); r.add_argument('--out',required=True); r.add_argument('--expected-experiment',choices=('M02','M04'))
    a=p.parse_args()
    if a.expected_experiment: check(read(a.config if a.cmd=='freeze' else a.plan)['experiment']==a.expected_experiment,'Experiment does not match launcher')
    if a.cmd=='freeze': freeze(a.config,a.m01_protocol,a.out)
    else: report(a.plan,a.frames,a.timing,a.out)

if __name__=='__main__': main()
