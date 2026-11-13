#!/usr/bin/env python3
"""Y05-R: matched current-only C* vs current+H4 future CF training and evaluation."""
from y05r_common import *
from types import SimpleNamespace
import argparse, fcntl, hashlib, os, sys, time, traceback, zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INSTALLED = ('gt4a2_y05r.py', 'y05r_common.py', 'y05r_train.py', 'y05r_eval.py', 'test_y05r.py', 'y05r_identity.json', 'y05r_SHA256SUMS')


def prepare(args, out):
    identity = jread(HERE / 'y05r_identity.json')
    y05 = source_y05_dir(args.y05_dir, ROOT, identity)
    old_protocol_path = y05 / 'protocol.json'; old_plan = jread(old_protocol_path)
    for name, digest in identity['y05_source_sha256'].items():
        p = ROOT / name if name.endswith('.sh') else HERE / name
        require(p.is_file() and sha(p) == digest, 'Pinned original Y05 source missing/changed: ' + str(p))
    import y05_common as old
    baseargs = SimpleNamespace(**old_plan['paths'], smoke_only=False)
    ctx = old.preflight(baseargs)
    require(ctx.plan == old_plan, 'Original Y05 protocol/data/environment changed; run on the original Y05 machine and do not edit hashes.')
    ctx.out = y05; ctx.signature = sha(old_protocol_path)
    from y05_train import head_states
    states, selected = head_states(ctx)
    require(ctx.rt.exp1.state_sha256(states['C_current']) == identity['head_tensor_sha256']['C_current'],
            'Pinned C_current warmstart tensor SHA differs')
    require(selected['C_current']['sha256'] == sha(Path(selected['C_current']['path'])), 'C_current checkpoint file SHA mismatch')
    old_natural = {(r['config_id'], r['sequence'], int(r['index'])): r for r in cread(y05 / 'y05_frame_metrics.csv')}
    # Engine and all parent-bank generation must use C_current as the common frozen carrier head.
    ctx.head_state = states['C_current']
    ctx.y05_dir = y05; ctx.y05_states = states; ctx.y05_selected = selected; ctx.old_plan = old_plan
    ctx.old_natural = old_natural
    ctx.args = SimpleNamespace(**old_plan['paths'], resume=args.resume, y05_dir=str(y05), smoke_only=args.smoke_only)
    ctx.out = Path(out).resolve()
    settings = frozen_settings(args.smoke_only)
    samples = old.prefix_samples(ctx.manifests, args.smoke_only)
    order = old.sample_order(samples, settings['fixed_updates'], settings['seed'])
    eval_plan = old.probe_cases(args.smoke_only)
    # Carry the original audited inputs forward, then pin the Y05 result bundle and this package.
    inputs = dict(old_plan['input_sha256'])
    additional = [y05/'protocol.json', y05/'summary.json', y05/'y05_frame_metrics.csv',
                  y05/'training_summary.json', y05/'gradient_audit.json',
                  Path(selected['C_current']['path']), Path(selected['C_current']['path']).parent/'curve.csv',
                  ROOT/'run_y05.sh']
    for name in INSTALLED:
        additional.append(HERE/name)
    additional.append(ROOT/'run_y05r.sh')
    for p in additional:
        require(p.is_file(), 'Required pinned input missing: ' + str(p))
        inputs[str(p.resolve())] = sha(p)
    warmstart = dict(kind='C_current', tensor_sha256=ctx.rt.exp1.state_sha256(ctx.head_state),
                     checkpoint_file=str(Path(selected['C_current']['path']).resolve()),
                     checkpoint_file_sha256=sha(Path(selected['C_current']['path'])))
    plan = dict(experiment='Y05-R', version=VERSION, partial_smoke=args.smoke_only,
        paths=dict(y05_dir=str(y05), **old_plan['paths']), settings=settings,
        warmstart=warmstart, source_y05_signature=identity['source_protocol_sha256'],
        sample_plan=samples, sample_order=order, sample_order_sha256=canonical(order),
        eval_plan=eval_plan, input_sha256=inputs, environment=old_plan['environment'],
        data_split='training anchors only RGB0..255; tail frames are development evaluation, not independent test',
        fixed_comparison=dict(C_star='current frame only + anchor delta regularizer',
                              CF='current frame + mean available H1..H4 future frame losses + same anchor delta regularizer',
                              same_initialization=True, same_sample_order=True, same_parent_bank=True,
                              same_optimizer=True, same_update_count=True, future_weight=1.0,
                              future_GT_exposure_extra_for_CF=True, harm_penalty=False, direction_gate=False),
        decision_rule=dict(natural_tail_macro_AbsRel_CF_must_beat_C_star=True,
            minimum_AbsRel_gain='max(1e-4, 3x maximum repeated-inference macro spread)',
            crowd2_non_degradation=True, balloon2_gt0p01_harmed_future_frames_must_not_increase=True,
            CF_H4_increment_must_survive_on_own_CF_S8_history=True,
            no_automatic_Y11_pass=True),
        limitations=['Three previously used development videos only; no independent test set or multi-seed claim',
                     'The C/S8 training parent bank is frozen and off-policy',
                     'Equal updates are not equal FLOPs, CUDA time, GT exposure, or total wall time',
                     'Y05-R screen success does not establish CVPR novelty or final method validity'])
    ctx.plan = plan
    return ctx


def verify_inputs(plan):
    for path, digest in plan['input_sha256'].items():
        require(Path(path).is_file() and sha(path) == digest, 'Pinned input changed during Y05-R: ' + path)


def archive(ctx):
    out = ctx.out; archive_path = out/'y05r_review.zip'; tmp = out/'y05r_review.zip.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if not p.is_file() or p.suffix not in ('.csv','.json','.jsonl','.md'):
                continue
            rel = p.relative_to(out)
            if any(part in ('cache','own_parents') for part in rel.parts):
                continue
            if p.name == 'failure.json' or p.name.endswith('.tmp'):
                continue
            z.write(p, rel.as_posix())
        for name in INSTALLED:
            z.write(HERE/name, 'source/y05r/'+name)
        z.write(ROOT/'run_y05r.sh', 'source/y05r/run_y05r.sh')
        for name in jread(HERE/'y05r_identity.json')['y05_source_sha256']:
            path = ROOT/name if name.endswith('.sh') else HERE/name
            z.write(path, 'reference/y05/'+name)
        z.write(ctx.y05_dir/'summary.json', 'reference/y05/summary.json')
        z.write(ctx.y05_dir/'protocol.json', 'reference/y05/protocol.json')
    with zipfile.ZipFile(tmp) as z:
        require(z.testzip() is None, 'Y05-R review ZIP CRC failed')
    tmp.replace(archive_path)
    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--y05-dir', default=os.environ.get('Y05_DIR','auto'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args(); out = Path(args.out_dir).resolve()
    require(out.parent.name.startswith('gt4a2_y05r_'), 'Use an independent outputs/gt4a2_y05r_<timestamp>/ namespace')
    if args.resume:
        frozen_path = out/'protocol.json'
        require(frozen_path.is_file(), '--resume requires the formal protocol.json')
        frozen = jread(frozen_path)
        require(frozen.get('experiment') == 'Y05-R' and not frozen.get('partial_smoke'), 'Only a formal Y05-R run can resume')
        done_summary = out/'summary.json'; done_zip = out/'y05r_review.zip'
        if done_summary.is_file() and done_zip.is_file():
            done = jread(done_summary)
            if done.get('status') == 'Y05R_COMPLETE_NOT_METHOD_PASS' and done.get('signature') == sha(frozen_path):
                print('[ALREADY COMPLETE] Existing Y05-R output is complete; no rerun needed.', flush=True)
                return 0
        args.y05_dir = frozen['paths']['y05_dir']; args.smoke_only = False
    if args.smoke_only:
        require(not args.resume, 'Smoke cannot resume; use a fresh smoke output folder')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); phase = 'preflight'
    with (out/'.run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This exact Y05-R output directory is already running')
        try:
            ctx = prepare(args, out); plan = ctx.plan
            require(ctx.y05_dir not in (out, *out.parents), 'Output directory cannot be inside the original Y05 result')
            protocol_path = out/'protocol.json'
            if protocol_path.exists():
                require(args.resume and jread(protocol_path) == plan, 'Existing protocol differs; resume only with the original frozen protocol')
            else:
                require(not args.resume, 'Cannot resume without a frozen protocol')
                jwrite(protocol_path, plan)
            ctx.signature = sha(protocol_path)
            print(f'[PREFLIGHT PASS] {VERSION}; arm updates={plan["settings"]["fixed_updates"]}; warmstart=C_current; sample windows={len(plan["sample_plan"])}', flush=True)
            print(f'[PROTOCOL] {protocol_path}; signature={ctx.signature}', flush=True)
            if args.preflight_only:
                print('[PREFLIGHT ONLY] no optimizer updates or evaluation performed', flush=True)
                return 0
            uuid = plan['environment'].get('uuid') or plan['environment'].get('gpu_name') or 'unknown-gpu'
            gpu_lock_path = Path('/tmp')/('svrdepth_y05r_'+hashlib.sha256(str(uuid).encode()).hexdigest()[:16]+'.lock')
            with gpu_lock_path.open('a+') as gpu_lock:
                print('[GPU QUEUE] waiting for exclusive GPU access', flush=True)
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                ctx.rt.torch.set_num_threads(4)
                if args.resume:
                    # Verify every input before restoring optimizer state.
                    verify_inputs(plan)
                phase = 'training'
                from y05r_train import train_all
                train_all(ctx)
                phase = 'evaluation'
                from y05r_eval import evaluate_all
                evaluate_all(ctx)
                verify_inputs(plan)
                if args.smoke_only:
                    print('[SMOKE PASS] Real CUDA smoke completed. This is not a method result.', flush=True)
                    return 0
                phase = 'archive'
                result = archive(ctx)
                summary = jread(out/'summary.json'); summary['review_package'] = str(result.resolve())
                jwrite(out/'summary.json', summary)
                print('[VERIFIED]', result.resolve(), flush=True)
                print('[CHECKPOINTS]', out/'training/C_star/final.pt', out/'training/CF/final.pt', flush=True)
            with (out/'invocations.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(completed=True, resume=args.resume, wall_sec=time.perf_counter()-started, unix_time=time.time()), allow_nan=False)+'\n')
            return 0
        except BaseException as exc:
            if phase in ('evaluation','archive'):
                (out/'summary.json').unlink(missing_ok=True)
            jwrite(out/'failure.json', dict(phase=phase, error=str(exc), traceback=traceback.format_exc()))
            with (out/'invocations.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(completed=False, resume=args.resume, wall_sec=time.perf_counter()-started, unix_time=time.time()), allow_nan=False)+'\n')
            raise


if __name__ == '__main__':
    raise SystemExit(main())
