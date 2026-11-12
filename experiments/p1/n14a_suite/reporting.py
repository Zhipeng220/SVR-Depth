"""Equal-recording metrics, propagation-phase attribution and frozen G0-G7 screening."""
from .common import *
import statistics

def report(ctx,rows,times,smoke=False):
    np=ctx.rt.np;methods=['A','O_fast','O_multi','C_star',*ctx.policy['arms']]
    seqs=list(dict.fromkeys(r['sequence'] for r in rows));lookup={(r['method'],r['sequence'],int(r['index'])):r for r in rows}
    video=[];phases=[]
    keys=['AbsRel','RMSE','delta1','p95_AbsRel','harmed_fraction','severe_fraction','latency_ms']
    for method in methods:
        for seq in seqs:
            rr=[r for r in rows if r['method']==method and r['sequence']==seq and r['AbsRel']!=''];require(bool(rr),'No scored frames: '+seq)
            for r in rr:
                a=lookup['A',seq,int(r['index'])]
                require(a['AbsRel']!='' and int(r['valid_pixels'])==int(a['valid_pixels']) and r['mask_sha256']==a['mask_sha256'],'Methods changed GT mask/frame set')
            trials=[x for x in times if x['method']==method and x['sequence']==seq]
            require(len(trials)==(1 if smoke else ctx.policy['timing_repeats']) and all(x['verified'] for x in trials),'Incomplete or unverified timing')
            def stats(items):
                delta=[float(r['AbsRel'])-float(lookup['A',seq,int(r['index'])]['AbsRel']) for r in items]
                return dict(frames=len(items),AbsRel=statistics.mean(float(r['AbsRel']) for r in items),RMSE=statistics.mean(float(r['RMSE']) for r in items),delta1=statistics.mean(float(r['delta1']) for r in items),
                    p95_AbsRel=float(np.percentile([float(r['AbsRel']) for r in items],95)),harmed_fraction=sum(d>0 for d in delta)/len(delta),severe_fraction=sum(d>.01 for d in delta)/len(delta),severe_frames=sum(d>.01 for d in delta))
            video.append(dict(method=method,sequence=seq,**stats(rr),latency_ms=statistics.median(x['mean_ms'] for x in trials),
                median_frame_latency_ms=statistics.median(x['median_ms'] for x in trials),p95_frame_latency_ms=statistics.median(x['p95_ms'] for x in trials),
                mean_normalized_residual=statistics.mean(float(r['residual_norm']) for r in rr),micro_frames=sum(str(r['micro_applied']).lower()=='true' for r in rr)))
            for h in [*range(8),'1..7']:
                subset=[r for r in rr if (int(r['phase'])>0 if h=='1..7' else int(r['phase'])==h)]
                phases.append(dict(method=method,sequence=seq,phase=h,**stats(subset)) if subset else dict(method=method,sequence=seq,phase=h,frames=0,AbsRel='',RMSE='',delta1='',p95_AbsRel='',harmed_fraction='',severe_fraction='',severe_frames=0))
    csvwrite(ctx.out/'per_recording.csv',video);csvwrite(ctx.out/'phase_summary.csv',phases)
    groups={'old_tail':[s for s in seqs if s in ctx.old_sequences],'Y11_development':[s for s in seqs if s not in ctx.old_sequences]}
    macros=[];phase_macros=[]
    for group,ss in groups.items():
        if not ss:continue
        for method in methods:
            vv=[r for r in video if r['method']==method and r['sequence'] in ss]
            macros.append(dict(group=group,method=method,recordings=len(vv),**{k:statistics.mean(r[k] for r in vv) for k in keys}))
            for h in [*range(8),'1..7']:
                vv=[r for r in phases if r['method']==method and r['sequence'] in ss and r['phase']==h and r['frames']>0]
                phase_macros.append(dict(group=group,method=method,phase=h,recordings=len(vv),AbsRel=statistics.mean(r['AbsRel'] for r in vv) if vv else ''))
    csvwrite(ctx.out/'macro_summary.csv',macros);csvwrite(ctx.out/'phase_macro_summary.csv',phase_macros)
    audit=read(ctx.out/'state_invariance.json');require(audit['signature']==ctx.signature and audit['passed'],'State invariance audit failed/foreign')
    gates={};detail={};passed=False;phase_supported=False
    if not smoke:
        table={(r['group'],r['method']):r for r in macros};v={(r['method'],r['sequence']):r for r in video};ph={(r['group'],r['method'],str(r['phase'])):r for r in phase_macros};threshold=ctx.policy['thresholds']
        for group in groups:
            a0=table[group,'N14_A0'];a1=table[group,'N14_A1'];c=table[group,'C_star'];o=table[group,'O_multi'];p0=ph[group,'N14_A0','1..7'];p1=ph[group,'N14_A1','1..7']
            require(p0['recordings']==p1['recordings']==len(groups[group]),'Propagation-frame phase coverage incomplete')
            detail[group]=dict(A0_gain_over_Cstar=1-a0['AbsRel']/c['AbsRel'],A1_gain_over_Cstar=1-a1['AbsRel']/c['AbsRel'],quality_ratio_to_Omulti=a1['AbsRel']/o['AbsRel'],
                latency_ratio_to_Cstar=a1['latency_ms']/c['latency_ms'],propagation_phase_gain_A1_over_A0=1-p1['AbsRel']/p0['AbsRel'])
        detail['balloon_A1_severe']=v['N14_A1','balloon2']['severe_frames'];detail['balloon_Cstar_severe']=v['C_star','balloon2']['severe_frames']
        detail['person_severe_excess_over_Ofast']=v['N14_A1','person_tracking2']['severe_fraction']-v['O_fast','person_tracking2']['severe_fraction']
        detail['per_recording_delta_A1_minus_A0']={s:v['N14_A1',s]['AbsRel']-v['N14_A0',s]['AbsRel'] for s in seqs}
        gates=dict(G0_GT_coverage=all(r['passed'] for r in ctx.plan['coverage']),G1_state_and_smoke=audit['passed'] and read(ctx.out/'smoke_checks.json')['passed'],
            G2_gradient_isolation=read(ctx.out/'gradient_audit.json')['passed'],
            G3_quality=all(x['A1_gain_over_Cstar']>=threshold['min_gain_over_Cstar']-1e-12 for x in [detail[g] for g in groups]),
            G4_Omulti_quality=all(detail[g]['quality_ratio_to_Omulti']<=threshold['quality_ratio_to_Omulti'] for g in groups),
            G5_risk=detail['balloon_A1_severe']<=detail['balloon_Cstar_severe'] and detail['person_severe_excess_over_Ofast']<=threshold['max_person_severe_excess_over_Ofast'],
            G6_latency=all(detail[g]['latency_ratio_to_Cstar']<=threshold['max_latency_ratio_to_Cstar'] for g in groups),
            G7_phase_value=all(detail[g]['propagation_phase_gain_A1_over_A0']>=threshold['min_phase_gain_A1_over_A0']-1e-12 for g in groups))
        passed=all(gates.values());phase_supported=gates['G7_phase_value']
    summary=dict(signature=ctx.signature,experiment='N14-A',status='SMOKE_COMPLETE' if smoke else 'N14A_DEVELOPMENT_SUITE_COMPLETE',quality_units=len(methods)*len(seqs),timing_trials=len(times),all_timing_verified=all(x['verified'] for x in times),
        screening_gate_passed=bool(passed),phase_value_supported=bool(phase_supported),N14B_candidate_eligible=bool(passed),N14B_started=False,method_pass=False,final_blind_test=False,training_seeds=1,
        gates=gates,gate_details=detail,state_invariance_passed=audit['passed'],parameter_count_per_arm=9244,current_GT_only=True,future_GT_exposures=0,
        prediction_retention=ctx.plan['prediction_retention'],timing_scope='RGB read/decode/preprocess/DIS/native step/transient decoder/D2H/sync; state hashes and metrics excluded',
        interpretation='h0 outputs bypass micro; h1..7 transient outputs never become persistent state. All eight recordings are development data.')
    write(ctx.out/'summary.json',summary)
    lines=['# N14-A 时相感知零写回微读出：开发结果','',f'状态：{summary["status"]}；筛查通过：{passed}；完整方法通过：False；最终盲测：False。','',
        'A0/A1 均为 9,244 参数，只使用当前 GT；冻结 C* 的持久状态逐帧核验，状态哈希不进入正式计时。',
        '旧三录像尾段与已查看的 Y11 五录像均为开发数据。单种子结果不能直接当作 CVPR 主实验。','',
        '|分组|方法|宏 AbsRel|宏 RMSE|δ1|严重伤害率|延迟 ms|','|---|---|---:|---:|---:|---:|---:|']
    for r in macros:lines.append(f'|{r["group"]}|{r["method"]}|{r["AbsRel"]:.6f}|{r["RMSE"]:.6f}|{r["delta1"]:.6f}|{r["severe_fraction"]:.4%}|{r["latency_ms"]:.3f}|')
    lines+=['','## 冻结 G0–G7','']+[f'- {k}: {value}' for k,value in gates.items()]
    lines+=['','参数少不代表额外 Decoder 读出免费；全部额外读出成本已进入实际计时。','零写回保证状态隔离，不保证当前输出零伤害。',
        'h=0..7 与 h=1..7 的逐录像/宏指标见 phase_summary.csv 和 phase_macro_summary.csv。A1 相对 A0 的传播帧质量需在两组开发数据均改善至少 1%，才支持本轮相位贡献。',
        '如果 A0 有效但 A1 无独立增量，只能支持普通零写回读出；如果质量、风险或成本未过门槛，不启动 N14-B。即使全过，也不自动训练 N14-B 或宣称最终方法成立。',
        '训练更新/实际 GPU 与墙钟成本见 training_summary.json；共享缓存成本与原始图像/GT/特征/光流/掩码 SHA 见 cache_manifest.json。']
    (ctx.out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');return summary
