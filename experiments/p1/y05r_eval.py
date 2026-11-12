"""Y05-R natural-history and paired single-pulse H4 evaluation."""
from y05r_common import *
from y05_engine import Engine, ensure_bank, parent_for, detached_state, numpy_metrics, tsave
import hashlib, math


def load_candidate_states(ctx):
    torch = ctx.rt.torch; result = {}
    for arm in ARMS:
        path = ctx.out / 'training' / arm / 'final.pt'
        require(path.is_file(), 'Missing final trained checkpoint: ' + str(path))
        obj = torch.load(path, map_location='cpu', weights_only=False)
        require(obj['signature'] == ctx.signature and obj['experiment'] == 'Y05-R' and obj['kind'] == arm,
                'Candidate checkpoint protocol identity differs: ' + arm)
        require(ctx.rt.exp1.state_sha256(obj['state']) == obj['head_sha256'], 'Candidate checkpoint tensor SHA differs: ' + arm)
        result[arm] = dict(state=obj['state'], sha256=obj['head_sha256'], path=str(path))
    return result


def natural_replay(ctx, engine, arm, state_dict, seq, wanted, keep_parents=True, max_frames=None):
    """Replay candidate's own K32/S8 history from RGB0; optional CPU snapshots at anchor-1."""
    torch = ctx.rt.torch; engine.configure(seq); engine.head.load_state_dict(state_dict, strict=True)
    stop = N_RGB[seq] if max_frames is None else min(N_RGB[seq], int(max_frames))
    snapshots = {}; rows = []; state = previous = None; start = 0
    for i in range(stop):
        score = (i < 256 or i >= TAIL[seq]) if max_frames is None else True
        frame = engine.load(i, score)
        apply = (i % 32 != 0 and i % 8 == 0)
        state, pred = engine.eval_step(state, previous, frame, 32, 'S8', apply=apply)
        mm = numpy_metrics(ctx.rt.np, pred, frame)
        split = 'prefix' if i < 256 else ('tail' if i >= TAIL[seq] else 'middle')
        row = dict(arm=arm, sequence=seq, index=i, split=split, scored=frame['gt_np'] is not None,
                   gt_expected=matched(ctx.manifests[seq][i]), repair_applied=apply,
                   prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(), **mm)
        rows.append(row)
        if keep_parents and i in wanted:
            snapshots[i] = detached_state(state, 'cpu')
        previous = frame
        if i == 0 or (i + 1) % 200 == 0 or i == stop - 1:
            print(f'[NATURAL {arm}] {seq} {i+1}/{stop}', flush=True)
    if keep_parents:
        require(sorted(snapshots) == sorted(wanted), f'Own-history parent snapshots incomplete: {arm}/{seq}')
        directory = ctx.out / 'evaluation' / 'own_parents' / arm / seq
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / 'parents.pt'
        tsave(torch, p, dict(signature=ctx.signature, arm=arm, sequence=seq,
                             head_sha256=ctx.rt.exp1.state_sha256(state_dict), snapshots=snapshots))
        jwrite(directory / 'complete.json', dict(signature=ctx.signature, arm=arm, sequence=seq,
                head_sha256=ctx.rt.exp1.state_sha256(state_dict), indices=sorted(wanted),
                parent_sha256=sha(p), frames=stop))
    return rows


def metric_rows_by_key(rows):
    return {(r['arm'], r['sequence'], int(r['index'])): r for r in rows if r.get('scored') and r.get('AbsRel') not in ('', None)}


def mean_metric(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in ('', None)]
    return sum(vals) / len(vals) if vals else None


def baseline_rows(ctx, variant, seq, indices):
    result = []
    if variant == 'A_K32_g0_f1/raw':
        source = ctx.refs
        for i in indices:
            row = source.get((variant, seq, int(i)))
            if row is not None and truth(row.get('matched')):
                result.append(dict(sequence=seq, index=int(i), **{k: row[k] for k in ('AbsRel','RMSE','delta1','valid_pixels')}))
    else:
        source = ctx.old_natural
        for i in indices:
            row = source.get((variant, seq, int(i)))
            if row is not None and truth(row.get('matched')):
                result.append(dict(sequence=seq, index=int(i), **{k: row[k] for k in ('AbsRel','RMSE','delta1','valid_pixels')}))
    return {(r['sequence'], r['index']): r for r in result}


def natural_summary(ctx, natural_rows, repeat_rows):
    primary = metric_rows_by_key(natural_rows); repeat = metric_rows_by_key(repeat_rows)
    summaries = []; repeat_checks = []
    per_arm_seq = {}; repeat_per_arm_seq = {}
    for arm in ARMS:
        for seq in SEQS:
            keys = sorted(k for k, r in primary.items() if r['arm'] == arm and r['sequence'] == seq and r['split'] == 'tail')
            a = [primary[k] for k in keys]
            b = [repeat[k] for k in keys if k in repeat]
            require(bool(a) and len(a) == len(b), f'No matched tail frames or repeat mismatch: {arm}/{seq}')
            m = {k: mean_metric(a, k) for k in ('AbsRel','RMSE','delta1')}
            mr = {k: mean_metric(b, k) for k in ('AbsRel','RMSE','delta1')}
            per_arm_seq[arm, seq] = m; repeat_per_arm_seq[arm, seq] = mr
            repeat_checks.append(dict(arm=arm, sequence=seq, split='tail', frames=len(a),
                primary_macro_frame_AbsRel=m['AbsRel'], repeat_macro_frame_AbsRel=mr['AbsRel'],
                macro_absrel_repeat_delta=abs(m['AbsRel']-mr['AbsRel']),
                primary_prediction_hash_match=sum(primary[k]['prediction_sha256']==repeat[k]['prediction_sha256'] for k in keys),
                total_frames=len(keys)))
            for variant in ('A_K32_g0_f1/raw', 'C_current_K32_S8_g1'):
                base = baseline_rows(ctx, variant, seq, [k[2] for k in keys])
                shared = [k for k in keys if (k[1], k[2]) in base and int(base[(k[1], k[2])]['valid_pixels']) == int(primary[k]['valid_pixels'])]
                require(len(shared) == len(keys), f'Baseline GT mask/frame mismatch for {variant}/{arm}/{seq}')
                base_rows = [base[(k[1], k[2])] for k in shared]
                summaries.append(dict(variant=arm, sequence=seq, split='tail', frames=len(a),
                    AbsRel=m['AbsRel'], RMSE=m['RMSE'], delta1=m['delta1'],
                    comparison=variant, comparison_AbsRel=mean_metric(base_rows,'AbsRel'),
                    delta_AbsRel=m['AbsRel']-mean_metric(base_rows,'AbsRel'),
                    comparison_RMSE=mean_metric(base_rows,'RMSE'), delta_RMSE=m['RMSE']-mean_metric(base_rows,'RMSE'),
                    comparison_delta1=mean_metric(base_rows,'delta1'), delta_delta1=m['delta1']-mean_metric(base_rows,'delta1')))
    # Macro means are equally weighted by sequence, not by number of frames.
    for arm in ARMS:
        for variant in ('A_K32_g0_f1/raw', 'C_current_K32_S8_g1'):
            cand = [per_arm_seq[arm, s] for s in SEQS]
            base_vals = []
            for seq in SEQS:
                keys = sorted(k for k, r in primary.items() if r['arm']==arm and r['sequence']==seq and r['split']=='tail')
                base = baseline_rows(ctx, variant, seq, [k[2] for k in keys])
                base_vals.append({k: mean_metric(list(base.values()), k) for k in ('AbsRel','RMSE','delta1')})
            cm = {k: sum(float(x[k]) for x in cand)/len(cand) for k in ('AbsRel','RMSE','delta1')}
            bm = {k: sum(float(x[k]) for x in base_vals)/len(base_vals) for k in ('AbsRel','RMSE','delta1')}
            summaries.append(dict(variant=arm, sequence='MACRO', split='tail', frames=sum(1 for r in natural_rows if r['arm']==arm and r['split']=='tail' and r['scored']),
                AbsRel=cm['AbsRel'], RMSE=cm['RMSE'], delta1=cm['delta1'], comparison=variant,
                comparison_AbsRel=bm['AbsRel'], delta_AbsRel=cm['AbsRel']-bm['AbsRel'],
                comparison_RMSE=bm['RMSE'], delta_RMSE=cm['RMSE']-bm['RMSE'],
                comparison_delta1=bm['delta1'], delta_delta1=cm['delta1']-bm['delta1']))
    for arm in ARMS:
        primary_macro = sum(per_arm_seq[arm, seq]['AbsRel'] for seq in SEQS) / len(SEQS)
        repeat_macro = sum(repeat_per_arm_seq[arm, seq]['AbsRel'] for seq in SEQS) / len(SEQS)
        repeat_checks.append(dict(arm=arm, sequence='MACRO', split='tail', frames=sum(r['frames'] for r in repeat_checks if r['arm']==arm and r['sequence']!='MACRO'),
            primary_macro_frame_AbsRel=primary_macro, repeat_macro_frame_AbsRel=repeat_macro,
            macro_absrel_repeat_delta=abs(primary_macro-repeat_macro),
            primary_prediction_hash_match=sum(r['primary_prediction_hash_match'] for r in repeat_checks if r['arm']==arm and r['sequence']!='MACRO'),
            total_frames=sum(r['total_frames'] for r in repeat_checks if r['arm']==arm and r['sequence']!='MACRO')))
    repeat_deltas = [r['macro_absrel_repeat_delta'] for r in repeat_checks if r['sequence']=='MACRO']
    return summaries, repeat_checks, per_arm_seq, gate_tolerance(repeat_deltas)



def natural_risk_rows(ctx, natural_rows):
    """Per-frame harm relative to A32 on the exact matched tail GT frames."""
    result=[]
    indexed=metric_rows_by_key(natural_rows)
    for arm in ARMS:
        for seq in SEQS:
            keys=sorted(k for k,r in indexed.items() if r['arm']==arm and r['sequence']==seq and r['split']=='tail')
            deltas=[]
            for key in keys:
                row=indexed[key]
                ref=ctx.refs.get(('A_K32_g0_f1/raw',seq,int(key[2])))
                require(ref is not None and truth(ref.get('matched')), f'A32 risk reference missing: {seq}/{key[2]}')
                require(int(ref['valid_pixels'])==int(row['valid_pixels']), f'A32 risk mask mismatch: {seq}/{key[2]}')
                deltas.append(float(row['AbsRel'])-float(ref['AbsRel']))
            require(bool(deltas), f'No natural risk frames: {arm}/{seq}')
            result.append(dict(arm=arm,sequence=seq,n_frames=len(deltas),
                harmed_frames_vs_A=sum(d>0 for d in deltas),
                severe_harmed_frames_gt0p01=sum(d>0.01 for d in deltas),
                mean_delta_AbsRel_vs_A=sum(deltas)/len(deltas),worst_delta_AbsRel_vs_A=max(deltas)))
    return result

def load_own_parent(ctx, arm, seq, index):
    directory = ctx.out / 'evaluation' / 'own_parents' / arm / seq
    meta = jread(directory / 'complete.json'); path = directory / 'parents.pt'
    require(meta['signature'] == ctx.signature and meta['parent_sha256'] == sha(path), 'Own-history parent cache identity changed')
    obj = ctx.rt.torch.load(path, map_location='cpu', weights_only=False)
    require(obj['signature'] == ctx.signature and obj['arm'] == arm and obj['sequence'] == seq, 'Own-history parent cache mismatch')
    require(index in obj['snapshots'], 'Required own-history parent snapshot absent')
    return obj['snapshots'][index]


def run_single_pulse(ctx, engine, case, parent, arm, arm_state, parent_source):
    torch = ctx.rt.torch; engine.configure(case['sequence']); engine.head.load_state_dict(arm_state, strict=True)
    frames, statics = engine.context(case, horizon=4); rows = []
    for action in ('skip', 'repair'):
        state = detached_state(parent, 'cuda')
        for h in range(5):
            apply = (h == 0 and action == 'repair')
            state, pred = engine.eval_step(state, frames[h-1], frames[h], 32, 'S8', apply=apply, static=statics.get(h))
            mm = numpy_metrics(ctx.rt.np, pred, frames[h])
            rows.append(dict(parent_source=parent_source, arm=arm, action=action,
                case_id=case['case_id'], sequence=case['sequence'], anchor=case['anchor'], age=case['age'],
                h=h, index=case['anchor']+h, matched=frames[h]['gt_np'] is not None,
                repair_applied=apply, prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(), **mm))
        del state
    del frames, statics
    torch.cuda.empty_cache()
    return rows


def h4_aggregate(frame_rows):
    grouped = {}
    for row in frame_rows:
        grouped.setdefault((row['parent_source'],row['arm'],row['case_id'],row['sequence'],int(row['anchor'])),{}).setdefault(row['action'],{})[int(row['h'])] = row
    windows=[]
    for (source,arm,case_id,seq,anchor), actions in sorted(grouped.items()):
        skip=actions.get('skip',{}); repair=actions.get('repair',{})
        hs=[h for h in range(1,5) if h in skip and h in repair and truth(skip[h]['matched']) and truth(repair[h]['matched'])]
        if not hs: continue
        s_abs=[float(skip[h]['AbsRel']) for h in hs]; r_abs=[float(repair[h]['AbsRel']) for h in hs]
        s_rmse=[float(skip[h]['RMSE']) for h in hs]; r_rmse=[float(repair[h]['RMSE']) for h in hs]
        s_d1=[float(skip[h]['delta1']) for h in hs]; r_d1=[float(repair[h]['delta1']) for h in hs]
        per_frame_delta=[r_abs[i]-s_abs[i] for i in range(len(hs))]
        windows.append(dict(parent_source=source,arm=arm,case_id=case_id,sequence=seq,anchor=anchor,
            matched_future_frames=len(hs),skip_H4_AbsRel=sum(s_abs)/len(s_abs),repair_H4_AbsRel=sum(r_abs)/len(r_abs),
            repair_minus_skip_H4_AbsRel=sum(per_frame_delta)/len(per_frame_delta),
            skip_H4_RMSE=sum(s_rmse)/len(s_rmse),repair_H4_RMSE=sum(r_rmse)/len(r_rmse),
            repair_minus_skip_H4_RMSE=sum(r_rmse)/len(r_rmse)-sum(s_rmse)/len(s_rmse),
            skip_H4_delta1=sum(s_d1)/len(s_d1),repair_H4_delta1=sum(r_d1)/len(r_d1),
            repair_minus_skip_H4_delta1=sum(r_d1)/len(r_d1)-sum(s_d1)/len(s_d1),
            better_future_frames=sum(x<0 for x in per_frame_delta),worse_future_frames=sum(x>0 for x in per_frame_delta),
            harmed_future_frames_gt0p01=sum(x>0.01 for x in per_frame_delta)))
    summary=[]
    for source in sorted({r['parent_source'] for r in windows}):
        for arm in ARMS:
            for seq in SEQS:
                rows=[r for r in windows if r['parent_source']==source and r['arm']==arm and r['sequence']==seq]
                if not rows: continue
                summary.append(dict(parent_source=source,arm=arm,sequence=seq,n_windows=len(rows),
                    mean_repair_minus_skip_H4_AbsRel=mean_metric(rows,'repair_minus_skip_H4_AbsRel'),
                    mean_repair_minus_skip_H4_RMSE=mean_metric(rows,'repair_minus_skip_H4_RMSE'),
                    mean_repair_minus_skip_H4_delta1=mean_metric(rows,'repair_minus_skip_H4_delta1'),
                    better_windows=sum(float(r['repair_minus_skip_H4_AbsRel'])<0 for r in rows),
                    worse_windows=sum(float(r['repair_minus_skip_H4_AbsRel'])>0 for r in rows),
                    harmed_future_frames_gt0p01=sum(int(r['harmed_future_frames_gt0p01']) for r in rows),
                    future_frames_scored=sum(int(r['matched_future_frames']) for r in rows)))
            rows=[r for r in windows if r['parent_source']==source and r['arm']==arm]
            if rows:
                summary.append(dict(parent_source=source,arm=arm,sequence='MACRO',n_windows=len(rows),
                    mean_repair_minus_skip_H4_AbsRel=mean_metric(rows,'repair_minus_skip_H4_AbsRel'),
                    mean_repair_minus_skip_H4_RMSE=mean_metric(rows,'repair_minus_skip_H4_RMSE'),
                    mean_repair_minus_skip_H4_delta1=mean_metric(rows,'repair_minus_skip_H4_delta1'),
                    better_windows=sum(float(r['repair_minus_skip_H4_AbsRel'])<0 for r in rows),
                    worse_windows=sum(float(r['repair_minus_skip_H4_AbsRel'])>0 for r in rows),
                    harmed_future_frames_gt0p01=sum(int(r['harmed_future_frames_gt0p01']) for r in rows),
                    future_frames_scored=sum(int(r['matched_future_frames']) for r in rows)))
    return windows, summary



def verify_reference_replay(ctx, engine, eval_cases, max_frames=None):
    """Replay pinned C_current and compare every scored frame/hash to original Y05 output."""
    state = ctx.y05_states['C_current']
    rows = []
    for seq in SEQS:
        wanted = sorted({int(c['snapshot_index']) for c in eval_cases if c['sequence']==seq})
        current = natural_replay(ctx, engine, 'C_current_reference', state, seq, wanted, keep_parents=True, max_frames=max_frames)
        for row in current:
            if not row['scored']:
                continue
            ref = ctx.old_natural.get(('C_current_K32_S8_g1', seq, int(row['index'])))
            require(ref is not None and truth(ref.get('matched')), f'Original C_current reference row missing: {seq}/{row["index"]}')
            require(int(ref['valid_pixels']) == int(row['valid_pixels']), f'C_current GT mask changed: {seq}/{row["index"]}')
            err = max(abs(float(ref[k])-float(row[k])) for k in ('AbsRel','RMSE','delta1'))
            require(err <= 2e-5, f'C_current natural replay metric mismatch {err}: {seq}/{row["index"]}')
            if ref.get('prediction_sha256'):
                require(ref['prediction_sha256'] == row['prediction_sha256'], f'C_current prediction hash mismatch: {seq}/{row["index"]}')
        rows.extend(current)
    return rows


def assert_state_equal(torch, a, b, label):
    aa = [a[0], a[1], *a[2], a[3]]; bb = [b[0], b[1], *b[2], b[3]]
    require(len(aa)==len(bb), 'State tensor count differs: '+label)
    for i,(x,y) in enumerate(zip(aa,bb)):
        require(x.shape==y.shape and torch.equal(x.detach().cpu(),y.detach().cpu()),
                f'C_current common/own parent state differs at tensor {i}: {label}')


def verify_common_parent_bank(ctx, common_paths, eval_cases):
    """Require shared C/S8 cached parents to equal the independently replayed C_current history."""
    torch=ctx.rt.torch; checked=0
    for case in eval_cases:
        own=load_own_parent(ctx,'C_current_reference',case['sequence'],int(case['snapshot_index']))
        shared=detached_state(parent_for(ctx,common_paths,case), 'cpu')
        assert_state_equal(torch,own,shared,f'{case["sequence"]}/{case["anchor"]}')
        checked+=1
    require(checked==len(eval_cases) and checked>0, 'No shared/own C_current parent state checks completed')
    return checked

def evaluate_all(ctx):
    torch = ctx.rt.torch; candidates = load_candidate_states(ctx)
    engine = Engine(ctx, SEQS[0]); natural_rows=[]; repeat_rows=[]
    try:
        eval_cases = [c for c in ctx.plan['eval_plan'] if c['origin']=='S8']
        wanted_by_seq = {seq: sorted({int(c['snapshot_index']) for c in eval_cases if c['sequence']==seq}) for seq in SEQS}
        max_frames = 64 if ctx.plan['partial_smoke'] else None
        reference_rows = verify_reference_replay(ctx, engine, eval_cases, max_frames=max_frames)
        cwrite(ctx.out/'y05r_reference_replay.csv', reference_rows)
        for arm in ARMS:
            for seq in SEQS:
                primary = natural_replay(ctx, engine, arm, candidates[arm]['state'], seq, wanted_by_seq[seq], keep_parents=True, max_frames=max_frames)
                natural_rows.extend(primary)
                repeat = natural_replay(ctx, engine, arm, candidates[arm]['state'], seq, wanted_by_seq[seq], keep_parents=False, max_frames=max_frames)
                repeat_rows.extend(repeat)
        cwrite(ctx.out/'y05r_natural_frames.csv', natural_rows)
        cwrite(ctx.out/'y05r_repeat_frames.csv', repeat_rows)
        if ctx.plan['partial_smoke']:
            # Exercise a real single-pulse branch and cache integrity; no method gate in smoke.
            common_paths = ensure_bank(ctx, engine, eval_cases, 'eval_common')
            common_parent_checks = verify_common_parent_bank(ctx, common_paths, eval_cases)
            sample = eval_cases[0]
            parent = parent_for(ctx, common_paths, sample)
            rows = run_single_pulse(ctx, engine, sample, parent, 'CF', candidates['CF']['state'], 'common_C_S8')
            cwrite(ctx.out/'y05r_h4_frames.csv', rows)
            jwrite(ctx.out/'y05r_gate.json', dict(status='SMOKE_ONLY_NOT_A_METHOD_RESULT', Y11_pass=False,
                note='Smoke confirms orchestration only; no quality or method conclusion.'))
            jwrite(ctx.out/'summary.json', dict(experiment='Y05-R', version=VERSION, signature=ctx.signature,
                status='SMOKE_PASS', partial_smoke=True, training_performed=True, arms=list(ARMS),
                steps_per_arm=ctx.plan['settings']['fixed_updates'], Y11_pass=False, method_pass=False,
                smoke_pulse_branches=2, common_parent_state_checks=common_parent_checks))
            return
        natural_summary_rows, repeat_checks, per_arm_seq, tolerance = natural_summary(ctx, natural_rows, repeat_rows)
        risk_rows = natural_risk_rows(ctx, natural_rows)
        cwrite(ctx.out/'y05r_natural_summary.csv', natural_summary_rows)
        cwrite(ctx.out/'y05r_natural_risk.csv', risk_rows)
        cwrite(ctx.out/'y05r_repeat_checks.csv', repeat_checks)
        # Shared C/S8 parent bank for paired common-history pulse tests.
        common_paths = ensure_bank(ctx, engine, eval_cases, 'eval_common')
        common_parent_checks = verify_common_parent_bank(ctx, common_paths, eval_cases)
        frame_rows=[]
        for case in eval_cases:
            seq=case['sequence']; idx=int(case['snapshot_index'])
            common_parent=parent_for(ctx, common_paths, case)
            own_parents={arm:load_own_parent(ctx,arm,seq,idx) for arm in ARMS}
            sources=[('common_C_S8',common_parent),('own_C_star_S8',own_parents['C_star']),('own_CF_S8',own_parents['CF'])]
            for source,parent in sources:
                for arm in ARMS:
                    frame_rows.extend(run_single_pulse(ctx,engine,case,parent,arm,candidates[arm]['state'],source))
            del common_parent, own_parents
        cwrite(ctx.out/'y05r_h4_frames.csv',frame_rows)
        windows, h4_summary = h4_aggregate(frame_rows)
        cwrite(ctx.out/'y05r_h4_windows.csv',windows)
        cwrite(ctx.out/'y05r_h4_summary.csv',h4_summary)
        # Frozen gate: natural tail improvement beyond repeat variation; Crowd2 non-degradation;
        # own-history H4 benefit; and no increase in Balloon2 >0.01 harmed future frames.
        cstar_macro = next(r for r in natural_summary_rows if r['variant']=='C_star' and r['sequence']=='MACRO' and r['comparison']=='C_current_K32_S8_g1')
        cf_macro = next(r for r in natural_summary_rows if r['variant']=='CF' and r['sequence']=='MACRO' and r['comparison']=='C_current_K32_S8_g1')
        cstar_crowd2 = per_arm_seq['C_star','crowd2']['AbsRel']; cf_crowd2 = per_arm_seq['CF','crowd2']['AbsRel']
        natural_gain = float(cstar_macro['AbsRel']) - float(cf_macro['AbsRel'])
        quality_pass = natural_gain > tolerance
        crowd2_pass = cf_crowd2 <= cstar_crowd2 + tolerance
        h4_by={(r['parent_source'],r['arm'],r['sequence']):r for r in h4_summary}
        own_cf = h4_by.get(('own_CF_S8','CF','MACRO'))
        own_cf_cstar_action = h4_by.get(('own_CF_S8','C_star','MACRO'))
        common_cf = h4_by.get(('common_C_S8','CF','MACRO'))
        common_cstar = h4_by.get(('common_C_S8','C_star','MACRO'))
        own_history_pass = bool(own_cf and own_cf_cstar_action and common_cf and common_cstar
            and float(own_cf['mean_repair_minus_skip_H4_AbsRel']) < 0
            and float(own_cf['mean_repair_minus_skip_H4_AbsRel']) < float(own_cf_cstar_action['mean_repair_minus_skip_H4_AbsRel'])
            and float(common_cf['mean_repair_minus_skip_H4_AbsRel']) < float(common_cstar['mean_repair_minus_skip_H4_AbsRel']))
        natural_cstar_balloon = next(r for r in risk_rows if r['arm']=='C_star' and r['sequence']=='balloon2')
        natural_cf_balloon = next(r for r in risk_rows if r['arm']=='CF' and r['sequence']=='balloon2')
        natural_harm_pass = int(natural_cf_balloon['severe_harmed_frames_gt0p01']) <= int(natural_cstar_balloon['severe_harmed_frames_gt0p01'])
        h4_harm_checks=[]
        for source in ('common_C_S8','own_C_star_S8','own_CF_S8'):
            cstar = h4_by.get((source,'C_star','balloon2')); cf = h4_by.get((source,'CF','balloon2'))
            if cstar and cf:
                ok = int(cf['harmed_future_frames_gt0p01']) <= int(cstar['harmed_future_frames_gt0p01'])
                h4_harm_checks.append(dict(parent_source=source,C_star_harmed_frames=int(cstar['harmed_future_frames_gt0p01']),
                    CF_harmed_frames=int(cf['harmed_future_frames_gt0p01']),pass_no_increase=ok))
        h4_harm_pass = bool(h4_harm_checks) and all(x['pass_no_increase'] for x in h4_harm_checks)
        harm_pass = natural_harm_pass and h4_harm_pass
        gate_pass = quality_pass and crowd2_pass and own_history_pass and harm_pass
        gate = dict(experiment='Y05-R', status='SCREEN_PASS' if gate_pass else 'SCREEN_FAIL_OR_INCONCLUSIVE',
            Y11_pass=False, method_pass=False, frozen_tolerance_AbsRel=tolerance,
            repeat_inference_macro_deltas=[r['macro_absrel_repeat_delta'] for r in repeat_checks if r['sequence']=='MACRO'],
            C_star_tail_macro_AbsRel=float(cstar_macro['AbsRel']), CF_tail_macro_AbsRel=float(cf_macro['AbsRel']),
            CF_gain_vs_C_star_AbsRel=natural_gain, quality_increment_pass=quality_pass,
            crowd2_C_star_tail_AbsRel=cstar_crowd2, crowd2_CF_tail_AbsRel=cf_crowd2, crowd2_non_degradation_pass=crowd2_pass,
            own_history_H4_increment_pass=own_history_pass,
            balloon2_natural_C_star_severe_harmed_frames=int(natural_cstar_balloon['severe_harmed_frames_gt0p01']),
            balloon2_natural_CF_severe_harmed_frames=int(natural_cf_balloon['severe_harmed_frames_gt0p01']),
            balloon2_natural_harm_nonincrease_pass=natural_harm_pass, balloon2_H4_harm_checks=h4_harm_checks,
            balloon2_H4_harm_nonincrease_pass=h4_harm_pass, balloon2_harm_nonincrease_pass=harm_pass,
            limitations=['Only three development videos and one training seed',
                         'Parent-bank training uses frozen C/S8 histories and remains off-policy',
                         'CF has extra future-GT supervision; equal-update is not equal-compute',
                         'Passing this screen does not grant Y11 PASS or establish CVPR contribution'])
        jwrite(ctx.out/'y05r_gate.json',gate)
        report = f'''# Y05-R result summary\n\n- Training arms: C* (current-only) and CF (current + H4 future mean).\n- Updates per arm: {ctx.plan['settings']['fixed_updates']}; warmstart: pinned C_current.\n- Natural tail macro AbsRel: C*={float(cstar_macro['AbsRel']):.8f}, CF={float(cf_macro['AbsRel']):.8f}.\n- CF gain over C* (positive is better): {natural_gain:.8f}; frozen tolerance: {tolerance:.8f}.\n- Natural quality gate: {quality_pass}; Crowd2 non-degradation: {crowd2_pass}; H4 own-history increment: {own_history_pass}; Balloon2 natural severe-harm non-increase: {natural_harm_pass}; H4 pulse-harm non-increase: {h4_harm_pass}.\n- Overall Y05-R screen: {gate['status']}. Y11 PASS: **False** (this experiment is a go/no-go screen, not final method validation).\n\n## Interpretation limits\n\nOnly three development videos and one seed are included. Training parents are a shared frozen C/S8 off-policy bank. CF receives additional future GT supervision; equal update count does not mean equal compute or equal GT exposure. A passing screen only justifies an independent candidate confirmation and later M01-M06; it does not establish CVPR-level novelty or generalization.\n'''
        (ctx.out/'Y05R_RESULT_SUMMARY.md').write_text(report, encoding='utf-8')
        jwrite(ctx.out/'summary.json', dict(experiment='Y05-R', version=VERSION, signature=ctx.signature,
            status='Y05R_COMPLETE_NOT_METHOD_PASS', partial_smoke=False, training_performed=True, arms=list(ARMS),
            steps_per_arm=ctx.plan['settings']['fixed_updates'], natural_frame_records=len(natural_rows),
            repeat_frame_records=len(repeat_rows), natural_risk_records=len(risk_rows), h4_frame_records=len(frame_rows), h4_windows=len(windows),
            common_parent_state_checks=common_parent_checks, reference_replay_rows=len(reference_rows), gate_status=gate['status'], Y11_pass=False, method_pass=False, final_checkpoints={a:candidates[a]['path'] for a in ARMS}))
    finally:
        engine.close()
