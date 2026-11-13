#!/usr/bin/env python3
"""Paired Bonn recording/scene summaries; no timing-based success decisions."""
import argparse,csv,json,math,random,statistics
from collections import defaultdict
from pathlib import Path
from contract import digest,read_json,read_csv,require,unit_id

def q(values,p):
    a=sorted(values);x=(len(a)-1)*p;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)

def write_csv(p,rows):
    if not rows:return
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def average_optional(values):
    valid=[float(x) for x in values if x!='']
    return statistics.mean(valid) if valid else ''

def paired(a,b):
    require(len(a)==len(b),'Paired RGB coverage differs');d=[];matched=0;failed=0
    for x,y in zip(a,b):
        require(x['index']==y['index'] and x['gt_status']==y['gt_status'] and x['mask_sha256']==y['mask_sha256'] and x['valid_pixels']==y['valid_pixels'],'Paired frame/mask mismatch')
        if x['gt_status']!='MATCHED':continue
        matched+=1
        if x['AbsRel']=='' or y['AbsRel']=='':failed+=1;continue
        d.append(float(x['AbsRel'])-float(y['AbsRel']))
    return d,matched,failed

def boot(groups,seed=20261101,repeats=2000):
    values=list(groups.values());rng=random.Random(seed)
    if len(values)<3:return None
    draws=[statistics.mean(rng.choice(values) for _ in values) for _ in range(repeats)]
    return q(draws,.025),q(draws,.975)

def boot_ratios(groups,seed=20261101,repeats=2000):
    values=list(groups.values())
    if len(values)<3:return None
    rng=random.Random(seed);quality=[];latency=[]
    for _ in range(repeats):
        sample=[rng.choice(values) for _ in values]
        quality.append(statistics.mean(x[0] for x in sample)/statistics.mean(x[1] for x in sample))
        if all(x[2] is not None and x[3] is not None for x in sample):latency.append(statistics.mean(x[2] for x in sample)/statistics.mean(x[3] for x in sample))
    return dict(quality_low=q(quality,.025),quality_high=q(quality,.975),latency_low=q(latency,.025) if latency else '',latency_high=q(latency,.975) if latency else '')

def report(plan_file,quality_dir,out,timing_dir=None):
    plan=read_json(plan_file);quality=Path(quality_dir).resolve();result=read_json(quality/'summary.json')
    require(result['plan_sha256']==digest(plan_file),'Quality plan differs')
    require(result['m04_native_replay_sha256']==digest(quality/'m04_native_replay.json'),'M04 native replay audit changed')
    require(result['status'] in ('QUALITY_ONLY_COMPLETE','QUALITY_INCOMPLETE'),'Unknown quality run status')
    quality_complete=result['status']=='QUALITY_ONLY_COMPLETE'
    allrows={};missing=[];units={u['id']:u for u in plan['units']};seqs={s['sequence']:s for s in plan['sequences']}
    for seq in seqs:
        for uid in units:
            d=quality/'quality'/seq/uid
            if not (d/'complete.json').is_file():
                require(not quality_complete,'Completed run has missing unit');missing.append(seq+'/'+uid);continue
            require(result['completion_hashes'].get(seq+'/'+uid)==digest(d/'complete.json'),'Completion metadata changed')
            meta=read_json(d/'complete.json');require(meta['frames_sha256']==digest(d/'frames.csv'),'Frame table changed')
            expected={f'quality/{seq}/{uid}/depth/{i:06d}.npz' for i in range(seqs[seq]['RGB'])}
            require(set(meta['prediction_sha256'])==expected,'Retained prediction coverage incomplete')
            for p,h in meta['prediction_sha256'].items():require(digest(quality/p)==h,'Retained prediction changed')
            rr=read_csv(d/'frames.csv');require(len(rr)==seqs[seq]['RGB'],'Frame count changed');allrows[seq,uid]=rr
    timings={}
    if timing_dir:
        timing_dir=Path(timing_dir).resolve();ts=read_json(timing_dir/'summary.json')
        require(ts['status']=='TIMING_COMPLETE' and ts['plan_sha256']==digest(plan_file) and ts['quality_summary_sha256']==digest(quality/'summary.json'),'Timing/quality protocol mismatch')
        for seq in seqs:
            for uid in units:
                trials=[]
                for rep in (1,2,3):
                    p=timing_dir/seq/uid/f'r{rep}.json';key=seq+'/'+uid+f'/r{rep}'
                    require(ts['completion_hashes'].get(key)==digest(p),'Timing trial changed')
                    x=read_json(p);require(x['endpoint_clean'] is True and x['quality_probes']>0 and x['frames']==seqs[seq]['RGB'],'Unverified timing trial')
                    require(x['frames_sha256']==digest(timing_dir/seq/uid/f'r{rep}.frames.csv'),'Timing frame table changed')
                    trials.append(x)
                latency=statistics.median(t['mean_ms'] for t in trials)
                timings[seq,uid]=dict(latency_ms=latency,p50_ms=statistics.median(t['p50_ms'] for t in trials),p95_ms=statistics.median(t['p95_ms'] for t in trials),throughput_fps=1000/latency,peak_allocated_bytes=max(t['peak_allocated_bytes'] for t in trials))
    per=[];comparisons=[];m04_age=[];alt_risk=[]
    for seq,s in seqs.items():
        for uid,u in units.items():
            if u['experiment']=='M04':ref=f"M04__OFF__s{u['seed']}"
            else:
                refunit=next(x for x in units.values() if x['experiment']=='M01_M02' and x['family']=='A' and x['k']==u['k'])
                ref=refunit['id']
            rr=allrows.get((seq,uid));reference=allrows.get((seq,ref))
            if rr is None:
                per.append(dict(sequence=seq,scene=s['physical_scene_group'],unit=uid,experiment=u['experiment'],family=u['family'],seed=u['seed'],k=u['k'],unit_status='failed',matched='',valid_frames=0,paired_valid=0,paired_failed='',AbsRel='',RMSE='',delta1='',p95_AbsRel='',reference=ref,delta_absrel='',harmed_fraction='',severe_fraction='',temporal_valid_pairs='',temporal_depth_change_mae_m='',refresh_fraction='',latency_ms='',p50_ms='',p95_ms='',throughput_fps='',peak_allocated_bytes=''))
                continue
            delta,matched,failed=paired(rr,reference) if reference is not None else ([],sum(r['gt_status']=='MATCHED' for r in rr),'')
            valid=[r for r in rr if r['AbsRel']!='']
            tt=timings.get((seq,uid),dict(latency_ms='',p50_ms='',p95_ms='',throughput_fps='',peak_allocated_bytes=''))
            per.append(dict(sequence=seq,scene=s['physical_scene_group'],unit=uid,experiment=u['experiment'],family=u['family'],seed=u['seed'],k=u['k'],unit_status='complete' if reference is not None else 'reference_failed',matched=matched,valid_frames=len(valid),paired_valid=len(delta),paired_failed=failed,
                AbsRel=average_optional(r['AbsRel'] for r in valid),RMSE=average_optional(r['RMSE'] for r in valid),delta1=average_optional(r['delta1'] for r in valid),
                p95_AbsRel=q([float(r['AbsRel']) for r in valid],.95) if valid else '',reference=ref,delta_absrel=statistics.mean(delta) if delta else '',
                harmed_fraction=sum(x>0 for x in delta)/len(delta) if delta else '',severe_fraction=sum(x>.01 for x in delta)/len(delta) if delta else '',
                temporal_valid_pairs=sum(r.get('temporal_status')=='ok' for r in rr),temporal_depth_change_mae_m=average_optional(r.get('temporal_depth_change_mae_m','') for r in rr),
                refresh_fraction=sum(str(r.get('keyframe','')).lower()=='true' for r in rr)/len(rr) if 'keyframe' in rr[0] else '',**tt))
            if u['experiment']=='M04' and reference is not None and rr and 'age' in rr[0]:
                for age in range(16):
                    a=[r for r in rr if int(r['age'])==age];b=[r for r in reference if int(r['age'])==age]
                    if not a:continue
                    diff,matched_age,failed_age=paired(a,b)
                    m04_age.append(dict(sequence=seq,unit=uid,seed=u['seed'],arm=u['arm'],age=age,matched=matched_age,paired_valid=len(diff),paired_failed=failed_age,
                        AbsRel=average_optional(r['AbsRel'] for r in a),delta_vs_OFF=statistics.mean(diff) if diff else ''))
        for p in [u for u in units.values() if u['experiment']=='M01_M02' and u['family']=='P' and u['k']==16 and u['gain']==1 and u['frequency']==1]:
            for other in ('O_fast','O_multi'):
                candidates=[u for u in units.values() if u['experiment']=='M01_M02' and u['family']==other and u['k']==16 and u['gain']==1 and u['frequency']==1 and str(u['seed'])==str(p['seed'])]
                for o in candidates:
                    if (seq,p['id']) not in allrows or (seq,o['id']) not in allrows:continue
                    pr=allrows[seq,p['id']];orr=allrows[seq,o['id']];d,matched,failed=paired(pr,orr)
                    paired_values=[(float(a['AbsRel']),float(b['AbsRel'])) for a,b in zip(pr,orr) if a['gt_status']=='MATCHED' and a['AbsRel']!='' and b['AbsRel']!='']
                    tp=timings.get((seq,p['id']));to=timings.get((seq,o['id']))
                    comparisons.append(dict(sequence=seq,scene=s['physical_scene_group'],P_seed=p['seed'],reference=other,reference_seed=o['seed'],matched=matched,paired_valid=len(d),paired_failed=failed,
                        P_minus_reference_AbsRel=statistics.mean(d) if d else '',P_minus_reference_gt_001=sum(x>.01 for x in d)/len(d) if d else '',
                        P_paired_AbsRel=statistics.mean(x[0] for x in paired_values) if d else '',reference_paired_AbsRel=statistics.mean(x[1] for x in paired_values) if d else '',
                        P_latency_ms=tp['latency_ms'] if tp else '',reference_latency_ms=to['latency_ms'] if to else ''))
        alt_ids={x['id'] for x in plan.get('points',[]) if x['role']=='alternative'}
        for alt in [u for u in units.values() if u['experiment']=='M01_M02' and any(i in alt_ids for i in u['point_ids'])]:
            refs=[u for u in units.values() if u['experiment']=='M01_M02' and ((u['family']=='A' and u['k'] in (16,32)) or (u['family']=='O_fast' and u['k'] in (16,32)) or (u['family']=='P' and u['k']==16 and u['seed']==alt['seed']))]
            for ref in refs:
                if (seq,alt['id']) not in allrows or (seq,ref['id']) not in allrows:continue
                d,matched,failed=paired(allrows[seq,alt['id']],allrows[seq,ref['id']])
                alt_risk.append(dict(sequence=seq,scene=s['physical_scene_group'],alt_unit=alt['id'],alt_seed=alt['seed'],reference_unit=ref['id'],reference=ref['family'],reference_k=ref['k'],reference_seed=ref['seed'],matched=matched,paired_valid=len(d),paired_failed=failed,
                    delta_AbsRel=statistics.mean(d) if d else '',harmed_fraction=sum(x>0 for x in d)/len(d) if d else '',severe_fraction=sum(x>.01 for x in d)/len(d) if d else ''))
    groups=defaultdict(list)
    for r in per:groups[(r['unit'],r['scene'])].append(r)
    recordings=defaultdict(list)
    for r in per:recordings[r['unit']].append(r)
    recording_macro=[]
    for uid,rs in sorted(recordings.items()):
        if not quality_complete:continue
        recording_macro.append(dict(unit=uid,recordings=len(rs),**{k:average_optional(x[k] for x in rs) for k in ('AbsRel','RMSE','delta1','p95_AbsRel')},
            harmed_fraction=average_optional(x['harmed_fraction'] for x in rs),severe_fraction=average_optional(x['severe_fraction'] for x in rs),
            latency_ms=statistics.mean(x['latency_ms'] for x in rs) if timing_dir else ''))
    scenes=[]
    for (uid,scene),rs in sorted(groups.items()):
        if not quality_complete:continue
        scenes.append(dict(unit=uid,scene=scene,recordings=len(rs),AbsRel=average_optional(r['AbsRel'] for r in rs),
            RMSE=average_optional(r['RMSE'] for r in rs),delta1=average_optional(r['delta1'] for r in rs),p95_AbsRel=average_optional(r['p95_AbsRel'] for r in rs),
            harmed_fraction=average_optional(r['harmed_fraction'] for r in rs),severe_fraction=average_optional(r['severe_fraction'] for r in rs),
            latency_ms=statistics.mean(r['latency_ms'] for r in rs) if timing_dir else ''))
    all_verified=quality_complete and all(s['physical_scene_verified'] for s in seqs.values());intervals=[];scene_macros=[]
    if all_verified:
        grouped=defaultdict(list)
        for r in scenes:grouped[r['unit']].append(r)
        for uid,rs in sorted(grouped.items()):
            scene_macros.append(dict(unit=uid,scene_groups=len(rs),**{k:average_optional(x[k] for x in rs) for k in ('AbsRel','RMSE','delta1','p95_AbsRel')},
                harmed_fraction=average_optional(x['harmed_fraction'] for x in rs),severe_fraction=average_optional(x['severe_fraction'] for x in rs),
                latency_ms=statistics.mean(x['latency_ms'] for x in rs) if timing_dir else ''))
        keyed=defaultdict(lambda:defaultdict(list))
        for r in comparisons:
            if r['P_minus_reference_AbsRel']!='':keyed[(r['P_seed'],r['reference'],r['reference_seed'])][r['scene']].append(r['P_minus_reference_AbsRel'])
        for key,by_scene in sorted(keyed.items()):
            vals={scene:statistics.mean(items) for scene,items in by_scene.items()};ci=boot(vals)
            comps=[r for r in comparisons if (r['P_seed'],r['reference'],r['reference_seed'])==key and r['P_paired_AbsRel']!='']
            grouped_vals=defaultdict(list)
            for r in comps:grouped_vals[r['scene']].append(r)
            ratio_inputs={scene:(statistics.mean(x['P_paired_AbsRel'] for x in rs),statistics.mean(x['reference_paired_AbsRel'] for x in rs),
                statistics.mean(x['P_latency_ms'] for x in rs) if timing_dir else None,statistics.mean(x['reference_latency_ms'] for x in rs) if timing_dir else None) for scene,rs in grouped_vals.items()}
            ratio_ci=boot_ratios(ratio_inputs) if all(v[1]>0 for v in ratio_inputs.values()) else None
            rr=list(ratio_inputs.values()); quality_ratio=statistics.mean(x[0] for x in rr)/statistics.mean(x[1] for x in rr) if rr and statistics.mean(x[1] for x in rr)>0 else ''
            latency_ratio=statistics.mean(x[2] for x in rr)/statistics.mean(x[3] for x in rr) if rr and timing_dir else ''
            intervals.append(dict(P_seed=key[0],reference=key[1],reference_seed=key[2],scene_groups=len(vals),mean=statistics.mean(vals.values()),ci95_low=ci[0] if ci else '',ci95_high=ci[1] if ci else '',bootstrap_seed=20261101,bootstrap_repeats=2000))
            intervals[-1].update(quality_ratio=quality_ratio,quality_ratio_ci95_low=ratio_ci['quality_low'] if ratio_ci else '',quality_ratio_ci95_high=ratio_ci['quality_high'] if ratio_ci else '',
                latency_ratio=latency_ratio,latency_ratio_ci95_low=ratio_ci['latency_low'] if ratio_ci else '',latency_ratio_ci95_high=ratio_ci['latency_high'] if ratio_ci else '')
    dest=Path(out);require(not dest.exists(),'Report output already exists');dest.mkdir(parents=True)
    write_csv(dest/'per_recording.csv',per);write_csv(dest/'recording_macro.csv',recording_macro);write_csv(dest/'direct_P_comparisons.csv',comparisons);write_csv(dest/'M04_age_strata.csv',m04_age);write_csv(dest/'P_Alt_risk.csv',alt_risk)
    if all_verified:
        write_csv(dest/'per_scene.csv',scenes);write_csv(dest/'scene_macro.csv',scene_macros);write_csv(dest/'paired_scene_intervals.csv',intervals)
    write_json=dict(status='QUALITY_REPORT_ONLY' if quality_complete else 'PARTIAL_FAILURE_REPORT',role=plan['role'],plan_sha256=digest(plan_file),quality_summary_sha256=digest(quality/'summary.json'),missing_units=missing,failed_units=result.get('failed_units',[]),
        all_scene_groups_verified=all_verified,bootstrap_unit='physical scene group' if all_verified else ('not computed; quality units incomplete' if not quality_complete else 'not computed; scene identity unverified'),
        timing_included=bool(timing_dir),blindness=plan.get('blindness','UNKNOWN'),warning='Temporal metric is unwarped same-pixel GT depth-change MAE, not motion/visibility-corrected; no training cost or formal v2.1 decision. Estimates condition on valid paired frames; report all failures and coverage. Peak allocated GPU memory is not total device memory.')
    (dest/'summary.json').write_text(json.dumps(write_json,indent=2,ensure_ascii=False),encoding='utf-8');print('[REPORT]',dest,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--quality-dir',required=True);ap.add_argument('--timing-dir');ap.add_argument('--out-dir',required=True);a=ap.parse_args();report(a.plan,a.quality_dir,a.out_dir,a.timing_dir)
if __name__=='__main__':main()
