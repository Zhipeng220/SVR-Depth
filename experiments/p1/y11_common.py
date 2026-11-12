"""Y11-A fixed-checkpoint confirmation. No Torch dependency in protocol utilities."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math,random,statistics

VERSION='Y11-Fallback-1.0.1'
FILES=('y11_common.py','y11_data.py','gt4a2_y11_fallback.py','test_y11.py','y11_policy.json','y11_identity.json')
METHODS=('A','O_fast','O_multi','P')

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def canonical(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
def jread(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def jwrite(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');q.replace(p)
def cread(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cwrite(path,rows):
    require(bool(rows),'Cannot write empty table '+str(path));p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    with q.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    q.replace(p)
def truth(x):return str(x).lower() in ('true','1','yes')
def quantile(values,q):
    x=sorted(float(v) for v in values);require(bool(x),'Empty quantile');p=(len(x)-1)*q;i=int(p);j=min(i+1,len(x)-1)
    return x[i]+(p-i)*(x[j]-x[i])

def config(method):
    require(method in METHODS,'Unknown frozen method')
    return dict(id=f'{method}_K16_g{0 if method=="A" else 1}_f1',family=method,schedule='K16',gain=0. if method=='A' else 1.,frequency=1)

def expected_counts(method,n):
    k=(n+15)//16;f=n-k
    return dict(keyframe_calls=k,flow_calls=f,native_fast_calls=f,repair_head_calls=f if method=='P' else 0,
                output_adapter_calls=f if method.startswith('O_') else 0,additional_readout_passes=f if method=='O_multi' else 0,rejected_candidates=0)

def metric(np,pred,gt):
    mask=(gt>0)&np.isfinite(gt);require(bool(mask.any()),'GT-only mask is empty')
    p=pred[mask];g=gt[mask]
    require(bool((np.isfinite(p)&(p>0)).all()),'Invalid prediction on fixed GT mask; refusing to shrink mask')
    p=p.astype('float64');g=g.astype('float64')
    return dict(AbsRel=float(np.mean(abs(p-g)/g)),RMSE=float(np.sqrt(np.mean((p-g)**2))),
                delta1=float(np.mean(np.maximum(p/g,g/p)<1.25)),valid_pixels=int(mask.sum()),
                mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())

def aggregate(frame_rows,timing,policy):
    """Equal recording weight; repeats are timing repeats, never training seeds."""
    groups=defaultdict(list);base={}
    for r in frame_rows:
        if r['method']=='A' and truth(r['matched']):base[r['sequence'],int(r['index'])]=r
        if truth(r['matched']):groups[r['method'],r['sequence']].append(r)
    seqs=policy['confirmation_sequences'];seq_table=[]
    ledger={(r['method'],r['sequence'],int(r['index'])):r for r in frame_rows}
    require(len(ledger)==len(frame_rows),'Duplicate frame identities')
    for seq in seqs:
        reference={int(r['index']):r for r in frame_rows if r['method']=='A' and r['sequence']==seq}
        for method in METHODS:
            rows={int(r['index']):r for r in frame_rows if r['method']==method and r['sequence']==seq}
            require(set(rows)==set(reference),'RGB coverage differs across methods')
            require(all(truth(rows[i]['matched'])==truth(reference[i]['matched']) for i in reference),'GT selection differs across methods')
    severity=policy['confirmation_rules']['severe_frame_absolute_AbsRel_increase_vs_A']
    for method in METHODS:
        for seq in seqs:
            rr=groups[method,seq];require(bool(rr),'No scored frames '+method+'/'+seq)
            controls=[base[seq,int(r['index'])] for r in rr]
            require(all(int(r['valid_pixels'])==int(b['valid_pixels']) and r['mask_sha256']==b['mask_sha256'] for r,b in zip(rr,controls)),'GT masks differ across methods')
            delta=[float(r['AbsRel'])-float(b['AbsRel']) for r,b in zip(rr,controls)]
            trials=[t for t in timing if t['method']==method and t['sequence']==seq]
            require(len(trials)==policy['timing_repeats'],'Missing timing repeats')
            require(all(t['endpoint_clean'] and t['quality_checked'] for t in trials),'Unverified timing cannot enter report')
            seq_table.append(dict(method=method,sequence=seq,n_GT=len(rr),AbsRel=statistics.mean(float(r['AbsRel']) for r in rr),
                RMSE=statistics.mean(float(r['RMSE']) for r in rr),delta1=statistics.mean(float(r['delta1']) for r in rr),
                frame_AbsRel_p95=quantile([r['AbsRel'] for r in rr],.95),harmed_fraction=sum(d>1e-8 for d in delta)/len(delta),
                severe_fraction=sum(d>severity for d in delta)/len(delta),severe_frames=sum(d>severity for d in delta),
                latency_ms=statistics.median(t['whole_mean_ms'] for t in trials),latency_p95_ms=statistics.median(t['whole_p95_ms'] for t in trials),
                peak_allocated_gib=max(t['peak_allocated_gib'] for t in trials)))
    indexed={(r['method'],r['sequence']):r for r in seq_table};macro=[]
    for method in METHODS:
        rr=[indexed[method,s] for s in seqs]
        macro.append(dict(method=method,n_videos=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in ('AbsRel','RMSE','delta1','frame_AbsRel_p95','harmed_fraction','severe_fraction','latency_ms')},
            worst_video_AbsRel=max(r['AbsRel'] for r in rr),worst_video_relative_change_vs_A=max(r['AbsRel']/indexed['A',r['sequence']]['AbsRel']-1 for r in rr)))
    m={r['method']:r for r in macro};rules=policy['confirmation_rules'];p=m['P'];of=m['O_fast'];om=m['O_multi']
    checks=dict(minimum_videos=len(seqs)>=rules['minimum_videos'],
        quality_vs_fast=p['AbsRel']/of['AbsRel']<=rules['P_macro_AbsRel_ratio_to_O_fast_max'],
        latency_vs_fast=p['latency_ms']/of['latency_ms']<=rules['P_macro_latency_ratio_to_O_fast_max'],
        quality_vs_multi=p['AbsRel']/om['AbsRel']<=rules['P_macro_AbsRel_ratio_to_O_multi_max'],
        latency_vs_multi=p['latency_ms']/om['latency_ms']<=rules['P_macro_latency_ratio_to_O_multi_max'],
        worst_video_risk=p['worst_video_relative_change_vs_A']<=rules['P_worst_video_relative_AbsRel_increase_vs_A_max'],
        per_video_severe_risk=all(indexed['P',s]['severe_fraction']-indexed['O_fast',s]['severe_fraction']<=rules['P_per_video_severe_fraction_minus_O_fast_max'] for s in seqs))
    # Report all measured axes, with no rounding-based dominance or invented scalar score.
    dominance=[]
    for a in METHODS:
        for b in METHODS:
            if a==b:continue
            axes=('AbsRel','RMSE','frame_AbsRel_p95','harmed_fraction','severe_fraction','latency_ms','worst_video_AbsRel')
            weak=all(m[a][k]<=m[b][k] for k in axes) and m[a]['delta1']>=m[b]['delta1']
            strict=any(m[a][k]<m[b][k] for k in axes) or m[a]['delta1']>m[b]['delta1']
            dominance.append(dict(method=a,reference=b,observed_dominates=weak and strict))
    rng=random.Random(policy['bootstrap_seed']);intervals=[]
    for ref in ('A','O_fast','O_multi'):
        samples=[]
        for _ in range(policy['bootstrap_repeats']):
            draw=rng.choices(seqs,k=len(seqs));samples.append(statistics.mean(indexed['P',s]['AbsRel']-indexed[ref,s]['AbsRel'] for s in draw))
        intervals.append(dict(candidate='P',reference=ref,mean_paired_AbsRel_delta=statistics.mean(indexed['P',s]['AbsRel']-indexed[ref,s]['AbsRel'] for s in seqs),
            bootstrap_recording_95_low=quantile(samples,.025),bootstrap_recording_95_high=quantile(samples,.975),n_recordings=len(seqs),
            interpretation='descriptive recording-level bootstrap; same Bonn environment is not verified independent physical scenes'))
    decision=dict(prospective_checks=checks,working_point_supported_on_this_recording_set=all(checks.values()),
                  Y11_pass=False,reason='Y11-A one frozen training seed; no independent physical-scene/final-test or multi-seed certification',
                  training_seed_count=1,timing_repeats=policy['timing_repeats'],candidate_changed=False)
    return seq_table,macro,dominance,intervals,decision
