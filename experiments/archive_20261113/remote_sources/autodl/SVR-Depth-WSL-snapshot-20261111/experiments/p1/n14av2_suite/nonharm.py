"""Paired current-frame harm against C*, and recording-clustered descriptive intervals."""
from .common import *
import statistics

def nonharm_report(ctx,rows):
    by={(r['method'],r['sequence'],int(r['index'])):r for r in rows};frames=[];summary=[]
    for r in rows:
        if r['method'] not in ctx.policy['arms'] or r['AbsRel']=='':continue
        c=by['C_star',r['sequence'],int(r['index'])];require(c['AbsRel']!='' and r['mask_sha256']==c['mask_sha256'] and int(r['valid_pixels'])==int(c['valid_pixels']),'C* paired GT mask differs')
        delta=float(r['AbsRel'])-float(c['AbsRel'])
        frames.append(dict(method=r['method'],sequence=r['sequence'],index=int(r['index']),phase=phase(int(r['index'])),age32=int(r['index'])%32,frame_type=frame_type(int(r['index'])),current_AbsRel=float(r['AbsRel']),Cstar_AbsRel=float(c['AbsRel']),delta_AbsRel=delta,current_nonharm_hinge=max(0.,delta),worse_than_Cstar=delta>0,severe_vs_Cstar=delta>.01))
    for method,seq in dict.fromkeys((r['method'],r['sequence']) for r in frames):
        all_rows=[r for r in frames if r['method']==method and r['sequence']==seq]
        for category in ['all','K32_reset','S8_pulse','propagation']:
            rr=all_rows if category=='all' else [r for r in all_rows if r['frame_type']==category]
            if not rr:continue
            summary.append(dict(method=method,sequence=seq,category=category,frames=len(rr),mean_delta_AbsRel=statistics.mean(r['delta_AbsRel'] for r in rr),worst_frame_delta=max(r['delta_AbsRel'] for r in rr),mean_hinge=statistics.mean(r['current_nonharm_hinge'] for r in rr),worse_vs_Cstar_rate=sum(r['worse_than_Cstar'] for r in rr)/len(rr),severe_vs_Cstar_rate=sum(r['severe_vs_Cstar'] for r in rr)/len(rr)))
    csvwrite(ctx.out/'nonharm_frames.csv',frames);csvwrite(ctx.out/'nonharm_summary.csv',summary)
    macros=[]
    for group,sequences in [('old_tail',ctx.old_sequences),('Y11_development',ctx.policy['development_sequences'])]:
        for method in ctx.policy['arms']:
            for category in ['all','K32_reset','S8_pulse','propagation']:
                rr=[r for r in summary if r['method']==method and r['sequence'] in sequences and r['category']==category]
                if rr:macros.append(dict(group=group,method=method,category=category,recordings=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in ['mean_delta_AbsRel','mean_hinge','worse_vs_Cstar_rate','severe_vs_Cstar_rate']}))
    csvwrite(ctx.out/'nonharm_macro_summary.csv',macros)
    return summary

def paired_bootstrap(np,left,right,repeats=2000,seed=20261031):
    require(len(left)==len(right) and len(left)>0,'Paired recording arrays differ');a=np.asarray(left,float);b=np.asarray(right,float)
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(a),size=(repeats,len(a)));delta=(a[indices]-b[indices]).mean(1)
    gain=1-a[indices].mean(1)/b[indices].mean(1)
    return dict(recordings=len(a),mean_paired_AbsRel_delta=float((a-b).mean()),delta_low95=float(np.percentile(delta,2.5)),delta_high95=float(np.percentile(delta,97.5)),macro_gain=float(1-a.mean()/b.mean()),gain_low95=float(np.percentile(gain,2.5)),gain_high95=float(np.percentile(gain,97.5)),bootstrap_repeats=repeats,unit='paired recordings, descriptive CI; no frame-IID or independent physical-scene claim')

def bootstrap_report(ctx,video,phases):
    out=[]
    for group,sequences in [('old_tail',[s for s in ctx.old_sequences if any(r['sequence']==s for r in video)]),('Y11_development',[s for s in ctx.policy['development_sequences'] if any(r['sequence']==s for r in video)])]:
        if not sequences:continue
        for phase_label in ['all','1..7']:
            source=video if phase_label=='all' else [r for r in phases if r['phase']=='1..7'];table={(r['method'],r['sequence']):r for r in source}
            for method,reference in [('N14_A0','C_star'),('N14_A1','C_star'),('N14_A1','N14_A0')]:
                left=[float(table[method,s]['AbsRel']) for s in sequences];right=[float(table[reference,s]['AbsRel']) for s in sequences]
                out.append(dict(group=group,phase=phase_label,method=method,reference=reference,**paired_bootstrap(ctx.rt.np,left,right,ctx.policy['bootstrap_repeats'],ctx.policy['bootstrap_seed'])))
    csvwrite(ctx.out/'paired_recording_bootstrap.csv',out)
