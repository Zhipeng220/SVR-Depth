"""Y10 alone: explicit post-readout F x radius/confidence cache interventions."""
from pathlib import Path
from y10_core import *

EXPERIMENT='Y10';PREFIX='y10';VERSION='Y10-1.0'
ENTRY='gt4a2_y10_state_commit_factorial.py'
DESCRIPTION='Y10: same-snapshot, same-weight, same-current-output 2x2 cache commit diagnosis'
CELLS={'F0_RC0':(0,0),'F1_RC0':(1,0),'F0_RC1':(0,1),'F1_RC1':(1,1)}

def state_contract():
    return dict(boundary='after native fast fusion/head/readout at anchor t, before committing the t+1 input cache',
        parent='exactly one cloned carrier snapshot at t-1; current RGB/flow and weights shared',
        native='(rA,cA,FA,rays)',repaired='(rP,cP,FP,rays)',
        factorial='commit F from native/repaired independently of paired radius/confidence cache; rays and all other inputs fixed',
        emission='all four cells emit identical repaired depth at t; t is excluded from future outcome contrasts',
        mixed_cell_interpretation='F1_RC0 and F0_RC1 are explicit off-manifold do-interventions, not self-consistent native/deployable states',
        why_interface_legal='same frame/grid/dtype and positive finite radius/confidence; fixed-K next step explicitly accepts radius and four F tensors',
        confidence_scope='fixed-K GT2 fast_step consumes radius and F, NOT previous confidence; RC effect here is principally radius-cache effect, not a confidence effect',
        future='head disabled for all four factorial cells; native keyframe reset kept; continuous_P is a separate natural control',
        independent_of_Y03=True)

def plan_description():
    return dict(cells=CELLS,current_readout='identical repaired prediction for all cells; no current-error benefit in future contrasts',
        horizons=list(HORIZONS),origins=['A','P'],gain=1.,future_repair='disabled in factorial; continuous_P separately enabled',
        effects='positive = error reduction; F_main=.5[(E00-E10)+(E01-E11)]; RC_main=.5[(E00-E01)+(E10-E11)]; interaction=E10+E01-E00-E11',
        no_hyperparameter_selection=True,confidence='passive-field canary verifies no effect in this fixed-K interface')

def factorial_effects(e00,e10,e01,e11):
    return dict(F_main=.5*((e00-e10)+(e01-e11)),RC_main=.5*((e00-e01)+(e10-e11)),
        interaction=e10+e01-e00-e11,F_when_RC0=e00-e10,F_when_RC1=e01-e11,
        RC_when_F0=e00-e01,RC_when_F1=e10-e11,joint_gain=e00-e11)

def unit(ctx,engine,case,snapshot,folder):
    import y10_runtime as rtmod
    torch=ctx.rt.torch;np=ctx.rt.np
    frames,statics=rtmod.case_context(ctx,engine,case)
    parent=rtmod.clone_state(snapshot,'cuda');parent_sha=ctx.rt.exp1.state_sha256(rtmod.flat_state(parent))
    native,native_pred,_,_=engine.step(rtmod.clone_state(parent),frames[-1],frames[0],case['K'],case['origin'],0,1.,statics[0])
    repaired,shared_pred,proposal_calls,_=engine.step(rtmod.clone_state(parent),frames[-1],frames[0],case['K'],case['origin'],1,1.,statics[0])
    require(proposal_calls==1,'Repair proposal must call the frozen head exactly once')
    require(torch.equal(native[3],repaired[3]),'Factor proposal changed rays')
    for key in rtmod.flat_state(native):
        a,b=rtmod.flat_state(native)[key],rtmod.flat_state(repaired)[key]
        require(a.shape==b.shape and a.dtype==b.dtype and a.device==b.device,'Crossed states have incompatible geometry/dtype/device')
    states={name:rtmod.mix_components(native,repaired,*bits) for name,bits in CELLS.items()}
    require(rtmod.equal_state(torch,states['F0_RC0'],native),'00 endpoint does not equal native proposal')
    require(rtmod.equal_state(torch,states['F1_RC1'],repaired),'11 endpoint does not equal repaired proposal')
    for name,state in states.items():
        require(torch.equal(state[3],native[3]),'Rays must be fixed in every cell')
        require(all(x.data_ptr()!=y.data_ptr() for x,y in zip(rtmod.flat_state(state).values(),rtmod.flat_state(parent).values())),'State aliases frozen parent')
    # A real interface canary, not just a source-code claim: confidence is passive.
    canary=rtmod.clone_state(native);canary=(canary[0],canary[1]*1.5,canary[2],canary[3])
    step_a,pred_a,_,_=engine.step(rtmod.clone_state(native),frames[0],frames[1],case['K'],case['origin'],0,1.,statics.get(1))
    step_b,pred_b,_,_=engine.step(canary,frames[0],frames[1],case['K'],case['origin'],0,1.,statics.get(1))
    require(rtmod.equal_state(torch,step_a,step_b) and np.array_equal(pred_a,pred_b),'Previous confidence unexpectedly affects this fixed-K path; revise contract')
    # Interleave another branch, then replay 00 to detect hidden mutable model/cache leakage.
    engine.step(rtmod.clone_state(repaired),frames[0],frames[1],case['K'],case['origin'],0,1.,statics.get(1))
    again,pred_again,_,_=engine.step(rtmod.clone_state(native),frames[0],frames[1],case['K'],case['origin'],0,1.,statics.get(1))
    require(rtmod.equal_state(torch,step_a,again) and np.array_equal(pred_a,pred_again),'Branch order changed model outputs')
    rows=[];files={};audits={};mismatch={}
    for name,state in states.items():
        fbit,rcbit=CELLS[name];expected_by_F=repaired if fbit else native
        mismatch[name]=dict(off_manifold_intervention=fbit!=rcbit,
            radius_relative_mismatch_to_readout_parent=rtmod.norm_delta(torch,[state[0]],[expected_by_F[0]]),
            confidence_relative_mismatch_to_readout_parent=rtmod.norm_delta(torch,[state[1]],[expected_by_F[1]]))
        rr,ff,aa=rtmod.eval_branch(ctx,engine,case,folder,name,state,shared_pred,1,frames,statics,lambda h,i:(0,1.))
        for r in rr:r.update(commit_F=fbit,commit_RC=rcbit,off_manifold_intervention=fbit!=rcbit,current_output_shared=True)
        rows.extend(rr);files.update(ff);audits[name]=aa
    rr,ff,aa=rtmod.eval_branch(ctx,engine,case,folder,'continuous_P',repaired,shared_pred,1,frames,statics,
        lambda h,i:(1,1.),natural_control=case['origin']=='P')
    for r in rr:r.update(commit_F=1,commit_RC=1,off_manifold_intervention=False,current_output_shared=True)
    rows.extend(rr);files.update(ff);audits['continuous_P']=aa
    if case['origin']=='A':
        # 00 shares the native A *future state*, but deliberately has repaired current emission.
        rtmod.prefix_metrics_natural(ctx,engine,case,'F0_RC0',[r for r in rows if int(r['h'])>0])
    h0=[r for r in rows if int(r['h'])==0]
    require(all(r['valid_prediction']==h0[0]['valid_prediction'] for r in h0),'Shared current output validity differs')
    if h0[0]['matched'] and h0[0]['valid_prediction']:
        require(all(all(float(r[m])==float(h0[0][m]) for m in ('AbsRel','RMSE','delta1')) for r in h0),'Current emission differs between cells')
    lookup={(r['arm'],int(r['h'])):r for r in rows}
    for h in range(1,H+1):
        if case['anchor']+h<case['next_keyframe']:continue
        rr=[lookup[name,h] for name in CELLS]
        if all(r['valid_prediction'] for r in rr) and rr[0]['matched']:
            require(all(max(abs(float(r[m])-float(rr[0][m])) for m in ('AbsRel','RMSE','delta1'))<=1e-7 for r in rr),'Full native reset failed to erase factorial history')
    require(ctx.rt.exp1.state_sha256(rtmod.flat_state(parent))==parent_sha,'Factorial fork changed parent state')
    write_json(Path(folder)/'cache_contract_audit.json',dict(signature=ctx.signature,case_id=case['case_id'],
        component_mismatch=mismatch,current_output_sha256=__import__('hashlib').sha256(shared_pred.tobytes()).hexdigest(),
        confidence_passive_canary=True,branch_order_canary=True,
        common_proposal_native_fast_calls=2,common_proposal_head_calls=1,
        common_proposal_note='The common proposal is reused across logical cells. Per-cell anchor call=1 is NOT additive physical runtime work.'))
    return rtmod.commit_unit(ctx,folder,case,rows,files,
        dict(same_parent=True,same_current_emission=True,rays_fixed=True,endpoints_native=True,
            explicit_mixed_cache_API=True,mixed_cells_are_not_native_consistent_states=True,
            confidence_passive_canary=True,branch_order_canary=True,natural_controls_checked=True),
        dict(branch_audits=audits))

def effects_from_windows(windows):
    groups={}
    for r in windows:groups.setdefault((r['case_id'],r['horizon']),{})[r['arm']]=r
    effects=[]
    for (case_id,horizon),rr in groups.items():
        require(all(name in rr for name in CELLS),'Missing factorial cell')
        first=rr['F0_RC0'];ok=all(rr[name]['status']=='OK' for name in CELLS)
        row={k:first[k] for k in ('case_id','sequence','K','origin','anchor','age','horizon')}
        row.update(status='OK' if ok else 'NUMERICAL_FAILURE_OR_NO_GT',n_GT=first['n_GT'],confidence_independent_effect_not_identified=True)
        contrast_names=factorial_effects(0,0,0,0)
        for scope,key in [('mean','mean_AbsRel'),('cumulative','sum_AbsRel')]:
            cc=factorial_effects(*(float(rr[name][key]) for name in CELLS)) if ok else {k:'' for k in contrast_names}
            row.update({scope+'_'+k:v for k,v in cc.items()})
        # Report pre-reset gain interaction separately; whole-horizon averaging can dilute it.
        before=all(rr[name]['pre_reset_gain']!='' for name in CELLS) and ok
        row['n_pre_reset_GT']=first['n_pre_reset_GT']
        row['pre_reset_interaction']=float(rr['F1_RC1']['pre_reset_gain'])-float(rr['F1_RC0']['pre_reset_gain'])-float(rr['F0_RC1']['pre_reset_gain']) if before else ''
        effects.append(row)
    return effects

def execute(ctx,runtime):
    if ctx.args.stage in ('all','diagnose'):rows,metas=runtime.execution_units(ctx,'diagnose',ctx.plan['tail_cases'],unit)
    else:rows,metas=runtime.close_and_collect(ctx,'diagnose',ctx.plan['tail_cases'])
    grouped={}
    for r in rows:grouped.setdefault(r['case_id'],[]).append(r)
    windows=[];case_map={c['case_id']:c for c in ctx.plan['tail_cases']}
    for case_id,rr in grouped.items():
        c=case_map[case_id]
        for w in paired_windows(rr,'F0_RC0'):
            w.update(case_id=case_id,sequence=c['sequence'],K=c['K'],origin=c['origin'],anchor=c['anchor'],age=c['age']);windows.append(w)
    effects=effects_from_windows(windows)
    write_csv(ctx.out/'y10_paired_future_windows.csv',windows)
    write_csv(ctx.out/'y10_factorial_effects.csv',effects)
    write_csv(ctx.out/'y10_sequence_diagnosis.csv',group_windows(windows,('sequence','K','origin','arm','horizon')))
    aggregate=[];gg={}
    for e in effects:gg.setdefault((e['sequence'],e['K'],e['origin'],e['horizon']),[]).append(e)
    for keys,rr in sorted(gg.items()):
        ok=all(r['status']=='OK' for r in rr);a=dict(zip(('sequence','K','origin','horizon'),keys))
        a.update(n_cases=len(rr),failed_cases=sum(r['status']!='OK' for r in rr),status='OK' if ok else 'INCOMPLETE_OR_FAILED')
        for name in [k for k in rr[0] if k.startswith(('mean_','cumulative_'))]:a[name]=statistics.mean(float(r[name]) for r in rr) if ok else ''
        aggregate.append(a)
    write_csv(ctx.out/'y10_factorial_sequence_effects.csv',aggregate)
    require(all(len(read_csv(ctx.out/'diagnose'/c['case_id']/'frames.csv'))==5*(H+1) for c in ctx.plan['tail_cases']),'Planned factorial arm/RGB count incomplete')
    return dict(arms_per_case=5,frame_records=len(rows),factorial_cells=4,
        numerical_failure_branches=sum(a['failed'] for m in metas for a in m['branch_audits'].values()),
        confidence_independent_effect_identified=False,natural_control_replay_passed=True,
        same_current_output_verified=True,off_manifold_mixed_cells_explicit=True,
        claims='post-readout cache-intervention diagnosis; no new deployable method, risk PASS or Y11 admission')

def result_note():
    return '''# Y10 结果阅读顺序

先看数值失败计数与 cache_contract_audit.json，再看 y10_factorial_effects.csv 和逐视频聚合。
正的 F_main / RC_main 表示降低未来误差；正 interaction 表示联合改善大于两项单独改善之和。
四格当前帧都输出同一份修复深度，当前帧不进入未来效应。h4/8/16 与复位前效果均报告。
F1_RC0 / F0_RC1 是有明确定义的跨缓存干预，不能称为原生一致状态或可部署方法。
confidence 在当前固定 K 路径不被读取，实机 canary 验证这一点；RC 因素不识别独立 confidence 因果贡献。
A-origin 与 P-origin 分开解释，连续 P 仅作为自然部署对照，不能替代四格效应。
缺 GT 的 RGB 仍传播；数值失败不删帧、不缩 GT mask，也不通过只平均有效窗口隐藏。
旧三视频上的诊断完成不等于机制已证实，更不是 Y11 或 CVPR 最终通过。
review ZIP 不包含大型 float32 预测、GT、快照与权重；它们保留在服务器。
'''
