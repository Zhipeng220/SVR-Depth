"""Two-arm Y05-R training: current-only C* versus current+H4 CF."""
from y05r_common import *
from y05_engine import Engine, ensure_bank, parent_for, detached_state, tsave, depth_loss, valid_state
import json, math, random, time


def seed(ctx):
    rt = ctx.rt
    s = ctx.plan['settings']['seed']
    random.seed(s); rt.np.random.seed(s); rt.torch.manual_seed(s); rt.torch.cuda.manual_seed_all(s)


def rng(ctx):
    t = ctx.rt.torch
    return dict(py=random.getstate(), np=ctx.rt.np.random.get_state(), torch=t.get_rng_state(), cuda=t.cuda.get_rng_state_all())


def restore_rng(ctx, value):
    random.setstate(value['py']); ctx.rt.np.random.set_state(value['np'])
    ctx.rt.torch.set_rng_state(value['torch']); ctx.rt.torch.cuda.set_rng_state_all(value['cuda'])


def cpu_head(engine):
    return {k: v.detach().cpu().clone() for k, v in engine.head.state_dict().items()}


def pulse_forward_r(ctx, engine, case, parent, frames, statics, kind, checkpoint=True, detach_after_anchor=False):
    """Single anchor repair, then no future head calls. Future loss remains differentiable through state."""
    torch = ctx.rt.torch
    require(kind in ARMS, 'Unknown Y05-R arm: ' + str(kind))
    state = detached_state(parent, parent[0].device)
    reference_state = detached_state(parent, parent[0].device)
    current = None; future = []; reference_future = []; n_future_gt = 0; delta = None
    for h in range(5):
        frame = frames[h]
        state, reg = engine.tensor_step(
            state, frames[h-1], frame, case['K'], case['origin'],
            apply=(h == 0), static=statics.get(h), checkpoint=checkpoint, capture=(h == 0))
        require(valid_state(torch, state), 'Nonfinite state during Y05-R training, including missing-GT RGB')
        if h == 0:
            delta = reg
            if detach_after_anchor:
                state = detached_state(state)
        if frame['gt'] is not None:
            pred = engine.read_depth(state)
            loss, _ = depth_loss(torch, pred, frame['gt'])
            if h == 0:
                current = loss
            else:
                future.append(loss); n_future_gt += 1
        with torch.no_grad():
            reference_state, _ = engine.tensor_step(
                reference_state, frames[h-1], frame, case['K'], case['origin'],
                apply=False, static=statics.get(h), checkpoint=False, capture=False)
            if h > 0 and frame['gt'] is not None:
                ref_loss, _ = depth_loss(torch, engine.read_depth(reference_state), frame['gt'])
                reference_future.append(ref_loss.detach())
    require(current is not None, 'No current GT for Y05-R sample')
    require(bool(future), 'No matched future GT in H4 Y05-R sample')
    require(delta is not None, 'Missing anchor delta regularizer')
    future_mean = torch.stack(future).mean()
    reference_mean = torch.stack(reference_future).mean() if reference_future else future_mean.detach().new_zeros(())
    mu = float(ctx.plan['settings']['mu_anchor_delta'])
    objective = current + (future_mean if kind == 'CF' else 0.0) + mu * delta
    return dict(objective=objective, current=current, future=future_mean,
                reference=reference_mean, delta=delta, future_losses=future,
                current_GT=1, future_GT=n_future_gt,
                objective_current_weight=1.0, objective_future_weight=1.0 if kind == 'CF' else 0.0)


def gradient_vector(torch, loss, params, retain=False):
    require(loss.requires_grad, 'CF future objective has no gradient graph')
    g = torch.autograd.grad(loss * 1024.0, params, allow_unused=True, retain_graph=retain)
    vector = torch.cat([(torch.zeros_like(p) if a is None else a).detach().float().reshape(-1)
                        for p, a in zip(params, g)]) / 1024.0
    require(bool(torch.isfinite(vector).all().item()) and float(vector.abs().sum().item()) > 0,
            'Future loss cannot reach C_current head with finite nonzero gradients')
    return vector


def gradient_audit(ctx, engine, paths):
    path = ctx.out / 'gradient_audit.json'
    if path.exists():
        obj = jread(path)
        require(obj['signature'] == ctx.signature and obj['passed'], 'Y05-R gradient audit identity differs')
        return obj
    torch = ctx.rt.torch; records = []
    engine.head.load_state_dict(ctx.head_state, strict=True)
    choices = {}
    for case in ctx.plan['sample_plan']:
        choices.setdefault((case['sequence'], case['origin']), case)
    wanted = list(choices) if not ctx.plan['partial_smoke'] else [next(iter(choices))]
    for seq, origin in wanted:
        case = choices[(seq, origin)]
        frames, statics = engine.context(case, horizon=4)
        parent = parent_for(ctx, paths, case); params = list(engine.head.parameters())
        result = pulse_forward_r(ctx, engine, case, parent, frames, statics, 'CF', checkpoint=True)
        g = gradient_vector(torch, result['future'], params, retain=True)
        grad_l1 = float(g.abs().sum().item())
        matched_future_gt = int(result['future_GT'])
        # Release the graph before running the detached negative control.
        del result, g
        torch.cuda.empty_cache()
        negative = pulse_forward_r(ctx, engine, case, parent, frames, statics, 'CF', checkpoint=True, detach_after_anchor=True)
        require(not negative['future'].requires_grad, 'Detached-state negative control still reaches future loss')
        records.append(dict(sequence=seq, origin=origin, case_id=case['case_id'],
                            future_gradient_l1=grad_l1, detach_negative_control=True,
                            matched_future_GT=matched_future_gt))
        del negative, frames, statics, parent
        torch.cuda.empty_cache()
    jwrite(path, dict(signature=ctx.signature, passed=True, records=records,
                      future_gradient_checked=True, detached_negative_control_checked=True,
                      scope='Real model/CUDA; no optimizer updates; one pre-registered canary per sequence/origin'))
    return jread(path)


def materialize(ctx, directory, kind, step, state, head_sha, elapsed_cuda):
    payload = dict(signature=ctx.signature, experiment='Y05-R', kind=kind, selection='final' if step == ctx.plan['settings']['fixed_updates'] else 'latest_recovery',
                   step=step, state=state, head_sha256=head_sha,
                   warmstart_checkpoint_sha256=ctx.plan['warmstart']['tensor_sha256'],
                   objective=ctx.plan['settings']['objective_' + kind])
    if step == ctx.plan['settings']['fixed_updates']:
        tsave(ctx.rt.torch, directory / 'final.pt', payload)
    cwrite(directory / 'curve.csv', ctx._active_curve)


def train_arm(ctx, engine, paths, kind):
    torch = ctx.rt.torch; cfg = ctx.plan['settings']; directory = ctx.out / 'training' / kind
    directory.mkdir(parents=True, exist_ok=True)
    recovery = directory / 'recovery.pt'; summary_path = directory / 'summary.json'; final_path = directory / 'final.pt'
    initial_state = ctx.head_state
    engine.head.load_state_dict(initial_state, strict=True)
    initial_sha = ctx.rt.exp1.state_sha256(cpu_head(engine))
    require(initial_sha == ctx.plan['warmstart']['tensor_sha256'], 'Warmstart is not the pinned C_current head')
    if summary_path.is_file() and final_path.is_file():
        summary = jread(summary_path); payload = torch.load(final_path, map_location='cpu', weights_only=False)
        require(summary['signature'] == ctx.signature and payload['signature'] == ctx.signature and payload['head_sha256'] == ctx.rt.exp1.state_sha256(payload['state']), 'Completed arm identity differs')
        print(f'[REUSE {kind}] final update already complete', flush=True)
        return summary
    seed(ctx)
    optimizer = torch.optim.AdamW(engine.head.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=cfg['amp_initial_scale'], growth_interval=cfg['amp_growth_interval'])
    curve = []; step0 = 0; prior_peak = 0.0
    if recovery.exists():
        require(ctx.args.resume, f'{kind}: recovery exists; resume with --resume-latest')
        obj = torch.load(recovery, map_location='cpu', weights_only=False)
        require(obj['signature'] == ctx.signature and obj['kind'] == kind and obj['initial_sha'] == initial_sha
                and obj['sample_order_sha256'] == ctx.plan['sample_order_sha256'], 'Recovery identity/sample order differs')
        curve = obj['curve']; step0 = int(obj['step']); prior_peak = float(obj.get('peak_cuda_allocated_gib', 0.0))
        require(step0 == len(curve) and [r['step'] for r in curve] == list(range(1, step0+1)), 'Recovery cursor is inconsistent')
        engine.head.load_state_dict(obj['state'], strict=True); optimizer.load_state_dict(obj['optimizer']); scaler.load_state_dict(obj['scaler']); restore_rng(ctx, obj['rng'])
        print(f'[RESUME {kind}] committed update {step0}/{cfg["fixed_updates"]}', flush=True)
    elif ctx.args.resume:
        print(f'[RESUME {kind}] no recovery found; starting this arm from the frozen C_current warmstart', flush=True)
    cases = {c['case_id']: c for c in ctx.plan['sample_plan']}
    journal = directory / 'physical_updates.jsonl'
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    ctx._active_curve = curve

    def commit(step):
        state = cpu_head(engine); digest = ctx.rt.exp1.state_sha256(state)
        obj = dict(signature=ctx.signature, kind=kind, step=step, initial_sha=initial_sha,
                   sample_order_sha256=ctx.plan['sample_order_sha256'], state=state,
                   optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=rng(ctx), curve=curve,
                   peak_cuda_allocated_gib=max(prior_peak, torch.cuda.max_memory_allocated()/1024**3))
        tsave(torch, recovery, obj)
        cwrite(directory / 'curve.csv', curve)
        if step == cfg['fixed_updates']:
            materialize(ctx, directory, kind, step, state, digest, sum(r['cuda_sec'] for r in curve))

    for step in range(step0 + 1, cfg['fixed_updates'] + 1):
        case = cases[ctx.plan['sample_order'][step-1]]
        torch.cuda.synchronize(); event0 = torch.cuda.Event(enable_timing=True); event1 = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter(); event0.record(); torch.cuda.synchronize()
        frames, statics = engine.context(case, horizon=4); parent = parent_for(ctx, paths, case)
        prep = time.perf_counter() - wall_start; optimizer.zero_grad(set_to_none=True); calls0 = engine.gate.calls
        result = pulse_forward_r(ctx, engine, case, parent, frames, statics, kind, checkpoint=True)
        require(bool(torch.isfinite(result['objective']).item()), 'Nonfinite Y05-R objective')
        scaler.scale(result['objective']).backward(); scaler.unscale_(optimizer)
        gradients = [p.grad for p in engine.head.parameters() if p.grad is not None]
        require(gradients and all(bool(torch.isfinite(g).all().item()) for g in gradients), 'Missing/nonfinite Y05-R head gradient')
        head_ids = {id(p) for p in engine.head.parameters()}
        require(all(not p.requires_grad and p.grad is None for p in engine.model.parameters() if id(p) not in head_ids), 'Frozen backbone acquired gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(engine.head.parameters(), cfg['clip_grad_norm']).item())
        require(math.isfinite(norm) and (step != 1 or norm > 0), 'Invalid/zero gradient norm')
        old_scale = float(scaler.get_scale()); scaler.step(optimizer); scaler.update()
        require(float(scaler.get_scale()) >= old_scale, 'AMP skipped optimizer update; recovery not advanced')
        require(all(bool(torch.isfinite(p).all().item()) for p in engine.head.parameters()), 'Optimizer produced nonfinite weights')
        event1.record(); torch.cuda.synchronize(); cuda_sec = float(event0.elapsed_time(event1))/1000.0
        require(math.isfinite(cuda_sec) and cuda_sec > 0, 'Invalid CUDA timing interval')
        row = dict(step=step, case_id=case['case_id'], sequence=case['sequence'], origin=case['origin'], anchor=case['anchor'],
                   horizon=4, arm=kind, objective=float(result['objective'].detach().item()),
                   current_AbsRel=float(result['current'].detach().item()), future_H4_AbsRel=float(result['future'].detach().item()),
                   future_reference_AbsRel=float(result['reference'].detach().item()), normalized_anchor_delta=float(result['delta'].detach().item()),
                   gradient_norm=norm, current_GT=result['current_GT'], future_GT=result['future_GT'],
                   logical_head_calls=1, physical_head_forwards=engine.gate.calls-calls0, cuda_sec=cuda_sec,
                   preparation_wall_sec=prep, step_wall_sec=time.perf_counter()-wall_start,
                   objective_current_weight=1.0, objective_future_weight=1.0 if kind == 'CF' else 0.0)
        curve.append(row)
        with journal.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        ctx._active_curve = curve
        if step % cfg['checkpoint_interval'] == 0 or step == cfg['fixed_updates']:
            commit(step)
        if step == 1 or step % 10 == 0 or step == cfg['fixed_updates']:
            print(f'[TRAIN {kind}] {step}/{cfg["fixed_updates"]} current={row["current_AbsRel"]:.6f} future={row["future_H4_AbsRel"]:.6f} CUDA={sum(r["cuda_sec"] for r in curve):.1f}s', flush=True)
        engine.gate.delta = None
        del result, frames, statics, parent
    require(len(curve) == cfg['fixed_updates'], f'{kind}: fixed update budget incomplete')
    if not final_path.is_file():
        state = cpu_head(engine); digest = ctx.rt.exp1.state_sha256(state)
        materialize(ctx, directory, kind, cfg['fixed_updates'], state, digest, sum(r['cuda_sec'] for r in curve))
    final = torch.load(final_path, map_location='cpu', weights_only=False)
    require(final['signature'] == ctx.signature and final['step'] == cfg['fixed_updates'] and final['head_sha256'] != initial_sha, 'Final checkpoint invalid or unchanged')
    physical = []
    physical_lines = [line for line in journal.read_text().splitlines() if line.strip()]
    for i, line in enumerate(physical_lines):
        try:
            physical.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(physical_lines)-1:
                raise
            print(f'[WARN {kind}] ignored incomplete trailing physical journal line during recovery', flush=True)
    summary = dict(signature=ctx.signature, experiment='Y05-R', kind=kind, status='TRAINING_COMPLETE', steps=len(curve),
                   initial_head_sha256=initial_sha, final_head_sha256=final['head_sha256'], warmstart='C_current',
                   sample_order_sha256=ctx.plan['sample_order_sha256'], committed_cuda_sec=sum(r['cuda_sec'] for r in curve),
                   physical_cuda_sec=sum(float(r['cuda_sec']) for r in physical), physical_updates=len(physical),
                   committed_current_GT_exposures=sum(int(r['current_GT']) for r in curve),
                   committed_future_GT_available_for_diagnostics=sum(int(r['future_GT']) for r in curve),
                   committed_future_GT_supervision_exposures=(sum(int(r['future_GT']) for r in curve) if kind == 'CF' else 0),
                   objective=cfg['objective_' + kind], future_weight=1.0 if kind == 'CF' else 0.0,
                   current_target_retained=True, harm_penalty=False, direction_gate=False,
                   peak_cuda_allocated_gib=max(prior_peak, torch.cuda.max_memory_allocated()/1024**3),
                   step_wall_sec=sum(float(r['step_wall_sec']) for r in curve), current_invocation_sec=time.perf_counter()-started,
                   compute_claim='equal updates only; CF has additional future GT supervision; no equal-compute claim')
    jwrite(summary_path, summary)
    del optimizer, scaler; torch.cuda.empty_cache()
    return summary


def train_all(ctx):
    engine = Engine(ctx, SEQS[0])
    paths = ensure_bank(ctx, engine, ctx.plan['sample_plan'], 'fit')
    try:
        gradient_audit(ctx, engine, paths)
        summaries = {}
        for kind in ARMS:
            summaries[kind] = train_arm(ctx, engine, paths, kind)
        jwrite(ctx.out / 'training_summary.json', summaries)
        return summaries
    finally:
        engine.close()
