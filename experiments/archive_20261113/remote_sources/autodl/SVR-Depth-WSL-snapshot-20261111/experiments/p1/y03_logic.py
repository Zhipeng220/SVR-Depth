"""Y03 alone: repeated fixed evidence, propagation allocation, prefix-selected EMA."""
import statistics
from pathlib import Path
from y03_core import *

EXPERIMENT='Y03';PREFIX='y03';VERSION='Y03-1.0'
ENTRY='gt4a2_y03_repeat_propagate.py'
DESCRIPTION='Y03: frozen-head repetition / propagation diagnosis; independent from Y10'
BETAS=(0.,.25,.5,1.)
BUDGETS=(2,4,8)

def state_contract():
    return dict(boundary='pre-readout feature-repair hook of one native fast transition',
        repeats='one RGB, one previous state, one DIS flow, one residual/fusion; apply the same head q times before ONE native readout',
        repeated_update='F[j+1] = F[j] + beta * (P(F[j]) - F[j]); gain=1 inside P',
        propagation='native fixed-K transitions, preserving RGB order and native keyframe resets',
        future='same parent snapshot, same future RGB and origin-matched AMP flow policy',
        EMA='relaxed same-evidence feature iteration, not an unaligned temporal average of features from different images',
        beta_zero='identity-state control; head proposals still computed and discarded; cannot count as learned protection',
        independent_of_Y10=True)

def plan_description():
    return dict(prefix_selection=dict(betas=list(BETAS),q=4,origins=['A'],objective='equal-video mean future AbsRel over h1..16; no tail GT; ties choose smaller beta'),
        tail_arms=['A_no_more_repair','single_pulse']+[f'{kind}_q{q}' for kind in ('repeat','alternate','tuned_ema') for q in BUDGETS]+['continuous_P'],
        matched_budget='repeat/alternate/tuned_ema pairs share q total head calls through h16; h4/h8 counts are shown and may differ',
        gain=1.,beta_selection='one global beta, not selected per video, K, age or tail case',
        natural_failure_check='natural P-origin continuous_P must reproduce completed Y09 on all available metrics')

def relaxed_iterations(features,head,q,beta):
    """Pure list algebra also used to independently test the mathematical contract."""
    return repeat_features(features,head,q,beta)

def schedules(kind,q,k,anchor):
    applied=0
    def future(h,index):
        nonlocal applied
        if kind=='continuous':return 1,1.
        if kind=='alternate' and index%k!=0 and applied<q-1:
            applied+=1;return 1,1.
        return 0,1.
    return future

def budget_count(kind,q,k,anchor,horizon):
    if kind in ('repeat','tuned_ema'):return q
    if kind=='single':return 1
    if kind=='none':return 0
    eligible=sum((anchor+h)%k!=0 for h in range(1,horizon+1))
    return 1+(eligible if kind=='continuous' else min(q-1,eligible))

def unit(ctx,engine,case,snapshot,folder,beta=None,selection=False):
    import y03_runtime as rtmod
    torch=ctx.rt.torch
    frames,statics=rtmod.case_context(ctx,engine,case)
    parent=rtmod.clone_state(snapshot,'cuda');parent_sha=ctx.rt.exp1.state_sha256(rtmod.flat_state(parent))
    native,pred0,_,_=engine.step(rtmod.clone_state(parent),frames[-1],frames[0],case['K'],case['origin'],0,1.,statics[0])
    sham,sham_pred,sham_calls,_=engine.step(rtmod.clone_state(parent),frames[-1],frames[0],case['K'],case['origin'],1,0.,statics[0])
    require(rtmod.equal_state(torch,native,sham) and ctx.rt.np.array_equal(pred0,sham_pred),'Zero-beta sham must reproduce native state/output exactly')
    arms=[dict(arm='A_no_more_repair',kind='none',q=0,beta=1.)]
    if selection:
        arms.extend(dict(arm=f'ema_beta_{b:g}',kind='tuned_ema',q=4,beta=b) for b in BETAS)
    else:
        arms.append(dict(arm='single_pulse',kind='single',q=1,beta=1.))
        for kind in ('repeat','alternate','tuned_ema'):
            for q in BUDGETS:arms.append(dict(arm=f'{kind}_q{q}',kind=kind,q=q,beta=beta if kind=='tuned_ema' else 1.))
        arms.append(dict(arm='continuous_P',kind='continuous',q=1,beta=1.))
    rows=[];files={};traces=[];audits={}
    for spec in arms:
        arm,kind,q,b=spec['arm'],spec['kind'],spec['q'],spec['beta']
        initial_q=q if kind in ('repeat','tuned_ema') else (0 if kind=='none' else 1)
        initial_beta=b if kind=='tuned_ema' else 1.
        start,pred,calls,trace=engine.step(rtmod.clone_state(parent),frames[-1],frames[0],case['K'],case['origin'],initial_q,initial_beta,statics[0])
        for t in trace:traces.append(dict(arm=arm,**t))
        natural=(case['origin']=='A' and kind=='none') or (case['origin']=='P' and kind=='continuous')
        rr,ff,aa=rtmod.eval_branch(ctx,engine,case,folder,arm,start,pred,calls,frames,statics,
            schedules(kind,q,case['K'],case['anchor']),natural_control=natural)
        if not aa['failed']:
            require(aa['head_calls']==budget_count(kind,q,case['K'],case['anchor'],H),'Equal-budget schedule changed')
        for r in rr:r.update(kind=kind,nominal_q=q,beta=b,matched_q_comparison=kind in ('repeat','alternate','tuned_ema'))
        rows.extend(rr);files.update(ff);audits[arm]=aa
        del start
    require(ctx.rt.exp1.state_sha256(rtmod.flat_state(parent))==parent_sha,'Fork mutated its frozen parent')
    # Alias checks compare all frames, not just aggregate quality.
    if selection or beta in (0.,1.):
        lookup={(r['arm'],r['h']):r for r in rows}
        if selection:pairs=[('ema_beta_0','A_no_more_repair')]
        elif beta==1.:pairs=[(f'tuned_ema_q{q}',f'repeat_q{q}') for q in BUDGETS]
        else:pairs=[(f'tuned_ema_q{q}','A_no_more_repair') for q in BUDGETS]
        for a,b in pairs:
            for h in range(H+1):
                x,y=lookup[a,h],lookup[b,h]
                require(x['valid_prediction']==y['valid_prediction'],'Alias validity differs')
                if x['matched'] and x['valid_prediction']:require(max(abs(float(x[m])-float(y[m])) for m in ('AbsRel','RMSE','delta1'))<=1e-7,'EMA boundary alias differs')
    write_csv(Path(folder)/'same_evidence_feature_trace.csv',traces)
    return rtmod.commit_unit(ctx,folder,case,rows,files,
        dict(parent_unchanged=True,zero_beta_sham=True,shared_RGB_and_flow=True,natural_controls_checked=True,
             first_intervention_shared=True,every_planned_RGB_recorded=True),
        dict(branch_audits=audits,selected_beta=beta,selection=selection))

def attach_case_windows(rows,case_map):
    grouped={}
    for r in rows:grouped.setdefault(r['case_id'],[]).append(r)
    output=[]
    for case_id,rr in grouped.items():
        case=case_map[case_id]
        for w in paired_windows(rr,'A_no_more_repair'):
            w.update(case_id=case_id,sequence=case['sequence'],K=case['K'],origin=case['origin'],anchor=case['anchor'],age=case['age'])
            output.append(w)
    return output

def select_beta(ctx,rows):
    require(all(r['split'] in ('prefix','smoke') and int(r['index'])<256 for r in rows),'Tail leakage in EMA selection')
    windows=attach_case_windows(rows,{c['case_id']:c for c in ctx.plan['prefix_cases']})
    candidates=[]
    for beta in BETAS:
        arm=f'ema_beta_{beta:g}';ww=[w for w in windows if w['arm']==arm and w['horizon']==16]
        valid=bool(ww) and all(w['status']=='OK' for w in ww)
        per_video={s:statistics.mean(float(w['mean_AbsRel']) for w in ww if w['sequence']==s) for s in SEQS} if valid else {}
        candidates.append(dict(beta=beta,status='OK' if valid else 'NUMERICAL_FAILURE_OR_NO_GT',
            prefix_macro_mean_future_AbsRel=statistics.mean(per_video.values()) if valid else '',per_video=per_video))
    eligible=[x for x in candidates if x['status']=='OK'];require(eligible,'No valid prefix EMA coefficient')
    chosen=min(eligible,key=lambda x:(x['prefix_macro_mean_future_AbsRel'],x['beta']))
    write_csv(ctx.out/'ema_prefix_scores.csv',[{k:v for k,v in x.items() if k!='per_video'} for x in candidates])
    selection=dict(signature=ctx.signature,selected_beta=chosen['beta'],q_for_tuning=4,horizon_for_tuning=16,
        split='training prefixes only',scope='one global beta; equal video weights; all K/ages/cycles within video equal',
        candidate_scores=candidates,unit_hashes={c['case_id']:digest(ctx.out/'prefix'/c['case_id']/'frames.csv') for c in ctx.plan['prefix_cases']},
        identity_selected=chosen['beta']==0.,repeat_alias_selected=chosen['beta']==1.)
    path=ctx.out/'ema_selection.json'
    if path.exists():require(read_json(path)==selection,'Previously frozen EMA selection changed')
    else:write_json(path,selection)
    return selection

def execute(ctx,runtime):
    if ctx.args.stage in ('all','select'):
        prefix_rows,_=runtime.execution_units(ctx,'prefix',ctx.plan['prefix_cases'],unit,selection=True)
    else:prefix_rows,_=runtime.close_and_collect(ctx,'prefix',ctx.plan['prefix_cases'])
    selection=select_beta(ctx,prefix_rows)
    if ctx.args.stage=='select':return None
    if ctx.args.stage in ('all','diagnose'):
        rows,metas=runtime.execution_units(ctx,'diagnose',ctx.plan['tail_cases'],unit,beta=selection['selected_beta'])
    else:rows,metas=runtime.close_and_collect(ctx,'diagnose',ctx.plan['tail_cases'])
    windows=attach_case_windows(rows,{c['case_id']:c for c in ctx.plan['tail_cases']})
    write_csv(ctx.out/'y03_paired_future_windows.csv',windows)
    grouped=group_windows(windows,('sequence','K','origin','arm','horizon'))
    write_csv(ctx.out/'y03_sequence_diagnosis.csv',grouped)
    contrasts=[];lookup={(r['case_id'],r['arm'],r['horizon']):r for r in windows}
    for case in ctx.plan['tail_cases']:
        for q in BUDGETS:
            a=lookup[case['case_id'],f'repeat_q{q}',16]
            for kind in ('alternate','tuned_ema'):
                b=lookup[case['case_id'],f'{kind}_q{q}',16]
                ok=a['status']==b['status']=='OK'
                if ok:require(a['head_calls']==b['head_calls']==q,'Matched-budget contrast has unequal realized calls')
                contrasts.append(dict(case_id=case['case_id'],sequence=case['sequence'],K=case['K'],origin=case['origin'],q=q,
                    comparison=kind+'_minus_repeat',status='OK' if ok else 'NUMERICAL_FAILURE_OR_NO_GT',
                    future_gain_difference=float(b['gain_vs_control'])-float(a['gain_vs_control']) if ok else '',
                    candidate_mean_AbsRel=b['mean_AbsRel'],repeat_mean_AbsRel=a['mean_AbsRel'],
                    positive_means='candidate has larger cumulative future gain than repeated same-evidence repair'))
    write_csv(ctx.out/'y03_equal_budget_contrasts.csv',contrasts)
    require(all(len(read_csv(ctx.out/'diagnose'/c['case_id']/'frames.csv'))==12*(H+1) for c in ctx.plan['tail_cases']),'Planned arm/RGB count incomplete')
    return dict(arms_per_case=12,frame_records=len(rows),prefix_cases=len(ctx.plan['prefix_cases']),
        selected_EMA_beta=selection['selected_beta'],EMA_identity_selected=selection['identity_selected'],
        numerical_failure_branches=sum(a['failed'] for m in metas for a in m['branch_audits'].values()),
        natural_control_replay_passed=True,matched_budget_contrasts=len(contrasts),
        claims='repetition/propagation mechanism diagnosis only; no trained new head, risk-control PASS or Y11 admission')

def result_note():
    return '''# Y03 结果阅读顺序

先看 summary.json 的数值失败分支，再看 y03_sequence_diagnosis.csv 与 y03_equal_budget_contrasts.csv。
future_gain_difference > 0 表示候选分配比同证据重复修复有更高未来累计收益。
只在 h16 比较 q 次修复预算；h4/h8 的实际调用数可能不同，不能冒充等预算。
EMA 系数只用原训练前缀选择，beta=0 是恒等，beta=1 与重复修复等价。
origin=A 与 origin=P 必须分开解释，P-origin 的零修复对照保留了此前的 P 历史。
连续 P 对照在 P-origin 必须复现 Y09 的自然尾段轨迹；该项与短窗口重复探针一起解释自然伤害。
三旧视频均为开发；抽取窗口按索引预定，不代表完整视频主结果或独立泛化。
GT、预测数组和快照留在服务器；review ZIP 只含 CSV/JSON/源码。完整深度保存可用 --save-depth all。
这些是诊断墙钟时间，不是实测部署耗时，不可并入 Y09 latency 排名。
'''
