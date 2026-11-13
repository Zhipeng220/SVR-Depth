"""M01 benchmark orchestration; real engine is injected only at execution."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from svr_depth.datasets.benchmark import iter_sequence
from svr_depth.evaluation.aggregate import summarize_frames, paired_risk, latency_summary, risk_hierarchy, latency_hierarchy
from svr_depth.evaluation.m01_metrics import array_digest, score_depth, score_ssi
from svr_depth.evaluation.m01_temporal import score_temporal_pair
from svr_depth.evaluation.paper_aggregate import summarize_coverage
from svr_depth.streaming.m01_engine import require_sha, sha256_file

ROOT = Path(__file__).resolve().parents[2]
METHODS = {'A', 'P_multi', 'O_fast', 'O_multi'}


def _json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')


def _csv(path, rows, columns=None):
    fields = list(dict.fromkeys((columns or []) + [k for r in rows for k in r]))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _path(value, config_dir):
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return str((expanded if expanded.is_absolute() else config_dir / expanded).resolve())


def load_config(path):
    path = Path(path).expanduser().resolve(strict=True)
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    if cfg.get('formal_paper_result') is True:
        raise ValueError('Formal paper output is unavailable until the author protocol and aggregate are verified')
    if 'development_frame_limit' in cfg:
        raise ValueError('Use the explicit --development-frames option, not a limit in the configuration')
    if cfg.get('refresh_interval') != 16 or cfg.get('gain') != 1 or cfg.get('frequency') != 1:
        raise ValueError('M01 requires refresh_interval=16, gain=1, frequency=1')
    temporal = cfg.get('temporal', {})
    if not isinstance(temporal, dict) or not isinstance(temporal.get('enabled', False), bool):
        raise ValueError('temporal must be a mapping with a boolean enabled flag')
    temporal_protocol = temporal.get('protocol', 'provisional_v1')
    if temporal.get('enabled') and temporal_protocol not in ('provisional_v1', 'paper_tau5_candidate_v1'):
        raise ValueError('Unknown temporal protocol')
    permitted_status = 'candidate' if temporal_protocol == 'paper_tau5_candidate_v1' else 'provisional'
    if temporal.get('protocol_status', permitted_status) != permitted_status:
        raise ValueError('Temporal scoring cannot be marked verified without author equivalence')
    threshold = temporal.get('fb_max_error_px')
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                  or not math.isfinite(threshold) or threshold <= 0):
        raise ValueError('fb_max_error_px must be a positive finite number')
    snap = cfg['snapshot']
    snap['path'] = _path(snap['path'], path.parent)
    for key in ('weights_sha256', 'config_sha256'):
        value = snap.get(key, '')
        if not re.fullmatch(r'[0-9a-fA-F]{64}', value):
            raise ValueError(f'snapshot.{key} must be the explicit 64-digit SHA256')
    methods = cfg.get('methods', [])
    if not methods or sum(x.get('method') == 'A' for x in methods) != 1:
        raise ValueError('methods must contain exactly one A baseline')
    seen = set()
    for m in methods:
        family = m.get('method')
        seed = str(m.get('seed', '')).strip()
        if family not in METHODS or not seed or (family, seed) in seen:
            raise ValueError('Unknown method, missing seed, or duplicate method/seed')
        m['seed'] = seed
        seen.add((family, seed))
        if family != 'A':
            if not m.get('checkpoint') or not re.fullmatch(r'[0-9a-fA-F]{64}', m.get('sha256', '')):
                raise ValueError(f'{family} requires checkpoint and explicit SHA256')
            m['checkpoint'] = _path(m['checkpoint'], path.parent)
            if 'state_key' not in m:
                raise ValueError(f'{family}: state_key must be explicit; null means raw state dict')
    sequences = cfg.get('sequences', [])
    if not sequences:
        raise ValueError('Configure at least one explicit sequence')
    seq_keys = set()
    for s in sequences:
        if s.get('frame_selection', 'full_sequence') not in ('full_sequence', 'release_last32'):
            raise ValueError('frame_selection must be full_sequence or release_last32')
        contract = s.get('temporal_inputs', {})
        if not isinstance(contract, dict):
            raise ValueError('temporal_inputs must be a mapping')
        for flag, evidence in (('flow_contract_verified','flow_evidence'),
                               ('camera_geometry_verified','camera_evidence'),
                               ('pose_provenance_verified','pose_evidence')):
            if flag in contract and not isinstance(contract[flag], bool):
                raise ValueError(f'{flag} must be a boolean')
            if contract.get(flag) and not str(contract.get(evidence, '')).strip():
                raise ValueError(f'{evidence} is required when {flag} is true')
        if contract.get('flow_contract_verified'):
            for key in ('flow_encoding','flow_geometry','fwd_field','bwd_field','fwd_mask_field',
                        'bwd_mask_field','fwd_index_offset','bwd_index_offset'):
                if key not in contract:
                    raise ValueError(f'verified flow contract requires {key}')
            if contract['flow_encoding'] != 'velodepth_packed12_rgb' or contract['flow_geometry'] != 'fraction_of_extent':
                raise ValueError('unsupported flow encoding or geometry')
            for key in ('fwd_index_offset','bwd_index_offset'):
                if isinstance(contract[key], bool) or not isinstance(contract[key], int):
                    raise ValueError(f'{key} must be an integer')
        for k in ('dataset', 'sequence', 'scene', 'depth_scale', 'min_depth_m', 'max_depth_m'):
            if k not in s or s[k] in ('', None):
                raise ValueError(f'Sequence missing {k}')
        if s['dataset'] not in ('Bonn', 'TUM', 'ScanNet', 'Sintel'):
            raise ValueError('M01 supports Bonn, TUM, ScanNet and Sintel')
        if (s['dataset'], s['sequence']) in seq_keys:
            raise ValueError('Duplicate dataset/sequence')
        seq_keys.add((s['dataset'], s['sequence']))
        if not all(math.isfinite(float(s[k])) for k in ('min_depth_m','max_depth_m','depth_scale')) or not 0 <= float(s['min_depth_m']) < float(s['max_depth_m']) or float(s['depth_scale']) <= 0:
            raise ValueError('Depth range and depth scale must be valid')
        if bool(s.get('pairs_csv')) == bool(s.get('hdf5')):
            raise ValueError('Specify exactly one of pairs_csv or hdf5')
        if s.get('pairs_csv'):
            if s['dataset'] != 'Bonn':
                raise ValueError('pairs_csv is supported only for raw Bonn')
            if 'max_delta_ms' not in s:
                raise ValueError('Bonn max_delta_ms must be explicit')
            s['pairs_csv'] = _path(s['pairs_csv'], path.parent)
        else:
            if not s.get('split_key'):
                raise ValueError('HDF5 split_key must be explicit')
            if s['dataset'] == 'ScanNet' and not s.get('sequences_key'):
                raise ValueError('ScanNet requires an explicit sequences_key; ScanNetVid uses sequences_.json')
            s['hdf5'] = _path(s['hdf5'], path.parent)
    return cfg


def _source_identity(cfg, config_path, mode):
    snapshot = Path(cfg['snapshot']['path'])
    config_file, config_sha = require_sha(snapshot/'config.json', cfg['snapshot']['config_sha256'])
    weights_file, weights_sha = require_sha(snapshot/'model.safetensors', cfg['snapshot']['weights_sha256'])
    checkpoints = []
    for m in cfg['methods']:
        if m['method'] != 'A':
            p, sha = require_sha(m['checkpoint'], m['sha256'])
            checkpoints.append({'method':m['method'], 'seed':m['seed'], 'path':p, 'sha256':sha})
    data_files = []
    done = set()
    hash_hdf5 = bool(cfg.get('hash_hdf5_content', False))
    for s in cfg['sequences']:
        p = Path(s['pairs_csv'] if s.get('pairs_csv') else s['hdf5']).resolve(strict=True)
        if p in done:
            continue
        done.add(p)
        st = p.stat()
        d = {'path': str(p), 'size_bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
             'sha256': sha256_file(p) if s.get('pairs_csv') or hash_hdf5 else None}
        data_files.append(d)
    source = {}
    for base in (ROOT/'svr_depth', ROOT/'velodepth', ROOT/'scripts', ROOT/'experiments/crosssource_v2/legacy_helpers'):
        for p in sorted(base.rglob('*.py')):
            source[str(p.relative_to(ROOT))] = sha256_file(p)
    configured = sorted({m['method'] for m in cfg['methods']})
    method_coverage = {'configured_methods':configured, 'missing_methods':sorted(METHODS - set(configured)),
                       'all_four_methods_configured':set(configured) == METHODS,
                       'training_seed_counts':{method:sum(m['method']==method for m in cfg['methods']) for method in sorted(METHODS - {'A'})}}
    return {'protocol':'M01 four-dataset K16 raw metric depth; not the upstream paper protocol', 'mode':mode, 'method_coverage':method_coverage,
            'backbone':{'config':config_file,'config_sha256':config_sha,
                        'weights':weights_file,'weights_sha256':weights_sha},
            'checkpoints':checkpoints, 'config_file':str(Path(config_path).resolve()),
            'config_sha256':sha256_file(config_path), 'resolved_config':cfg,
            'sources_sha256':source, 'data_files':data_files,
            'data_hash_scope':'Bonn pair CSV contents; HDF5 content only if hash_hdf5_content=true. Per-frame RGB/GT hashes in quality records.',
            'python':sys.version, 'platform':platform.platform(),
            'temporal_tau5':(cfg.get('temporal', {}).get('protocol', 'provisional_v1') + '; not author-verified' if cfg.get('temporal', {}).get('enabled') else 'not calculated'),
            'protocol_ids':{'A':'matched_K16_A', 'P_multi':'matched_K16_P', 'official_A':'separate upstream evaluator'},
            'upstream_depth_metric_scope':'optional upstream d1 and SSI functions on this runner mask; not the full upstream evaluation protocol',
            'metric':'raw camera-z; no scale or shift fitting', 'propagation_flow':'OpenCV DIS FAST; same FP32 flow boundary for all methods',
            'quality_and_timing_are_separate':True}


def _slug(s):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s))[:90] + '_' + hashlib.sha256(str(s).encode()).hexdigest()[:8]


def _write_progress(out, rows, calls, failures, temporal_rows=None):
    frame_columns = ['dataset','scene','sequence','method','seed','index','k','timestamp','age','keyframe',
                     'status','absrel','rmse','delta1','valid_pixels','mask_sha256','rgb_sha256','gt_sha256']
    _csv(out/'frame_metrics.csv', rows, frame_columns)
    _csv(out/'call_counts.csv', calls, ['dataset','scene','sequence','method','seed','status'])
    _csv(out/'failures.csv', failures, ['dataset','scene','sequence','method','seed','index','reason'])
    if temporal_rows is not None:
        _csv(out/'temporal_metrics.csv', temporal_rows,
             ['dataset','scene','sequence','method','seed','protocol_id','prev_index','curr_index',
              'prev_source_frame_id','curr_source_frame_id','status','issues','protocol_status',
              'tau5_provisional','tau5_ssi_provisional','valid_pixels','ssi_valid_pixels'])


def _write_quality(out, rows, calls, failures, bootstrap_samples, *, complete=True, development=False,
                   temporal_rows=None):
    if not complete or development:
        bootstrap_samples = 0
    _write_progress(out, rows, calls, failures, temporal_rows)
    seq, scene, dataset = summarize_frames(rows, bootstrap_samples=bootstrap_samples)
    _csv(out/'sequence_summary.csv', seq, ['dataset','scene','sequence','method','seed'])
    _csv(out/'scene_summary.csv', scene, ['dataset','scene','method','seed'])
    _json(out/'dataset_summary.json', {'source_run_status':'complete' if complete else 'incomplete', 'development_only':development, 'datasets':dataset})
    risks = paired_risk(rows)
    _csv(out/'risk_summary.csv', risks, ['dataset','scene','sequence','method','seed','reference'])
    risk_scene, risk_dataset, risk_combined = risk_hierarchy(risks, bootstrap_samples)
    _csv(out/'risk_scene_summary.csv', risk_scene, ['dataset','scene','method','seed','reference'])
    _csv(out/'risk_dataset_summary.csv', risk_dataset, ['dataset','method','seed','reference'])
    _json(out/'paired_comparison.json', {'source_run_status':'complete' if complete else 'incomplete', 'development_only':development, 'comparisons':risk_combined})
    if temporal_rows is not None:
        _json(out/'paper_coverage.json', summarize_coverage(rows, temporal_rows, development=development))


def _append_temporal(temporal_rows, previous, current, prefix, temporal_options):
    try:
        pair = score_temporal_pair(previous, current, temporal_options)
    except Exception as exc:
        pair = {'prev_index':previous['frame'].index, 'curr_index':current['frame'].index,
                'status':'temporal_exception', 'issues':'temporal_exception',
                'detail':f'{type(exc).__name__}: {exc}',
                'temporal_protocol':temporal_options.get('protocol', 'provisional_v1'),
                'protocol_status':'candidate' if temporal_options.get('protocol') == 'paper_tau5_candidate_v1' else 'provisional'}
    temporal_rows.append({**prefix, **pair})


def run_quality(cfg, out, engine_factory, base_sha, *, iterator_factory=iter_sequence):
    rows, calls, failures, identities = [], [], [], []
    temporal_options = cfg.get('temporal', {})
    temporal_rows = [] if temporal_options.get('enabled', False) else None
    incomplete = False
    runtime = None
    methods = sorted(cfg['methods'], key=lambda m: (m['method'] != 'A', m['method'], m['seed']))
    for method in methods:
        engine = engine_factory(cfg['snapshot']['path'], method, base_sha)
        if runtime is None and hasattr(engine, 'runtime_info'):
            runtime = engine.runtime_info()
        if getattr(engine, 'identity', None) is not None:
            identities.append({'method': method['method'], 'seed':method['seed'], **engine.identity})
        for spec in cfg['sequences']:
            engine.reset()
            prefix = {'dataset':spec['dataset'], 'scene':spec['scene'], 'sequence':spec['sequence'],
                      'method':method['method'], 'seed':method['seed'],
                      'protocol_id': {'A':'matched_K16_A', 'P_multi':'matched_K16_P'}.get(method['method'], 'legacy_K16')}
            status, index = 'complete', -1
            previous = None
            print(f"[M01 quality] {prefix}", flush=True)
            try:
                frames = iterator_factory(spec)
                if cfg.get('development_frame_limit') is not None:
                    frames = itertools.islice(frames, cfg['development_frame_limit'])
                for frame in frames:
                    index = frame.index
                    rgb_sha = array_digest(frame.rgb)
                    row = {**prefix, 'index':index, 'k':16, 'timestamp':frame.timestamp,
                           'source_frame_id':getattr(frame, 'source_frame_id', ''),
                           'source_frame_index':getattr(frame, 'source_frame_index', None),
                           'age':'', 'keyframe':'', 'status':'', 'absrel':'', 'rmse':'', 'delta1':'',
                           'valid_pixels':'', 'mask_sha256':'', 'rgb_sha256':rgb_sha, 'gt_sha256':''}
                    pred, meta, aligned = None, None, None
                    try:
                        prediction, meta = engine.step(index, frame.rgb)
                        pred = engine.to_numpy(prediction)
                        row.update(age=meta['age'], keyframe=meta['keyframe'])
                    except Exception as exc:
                        row['status'] = 'inference_failure'
                        failures.append({**prefix,'index':index,'stage':'inference','reason':f'{type(exc).__name__}: {exc}'})
                        if temporal_rows is not None and previous is not None:
                            _append_temporal(temporal_rows, previous, {'frame':frame, 'prediction':None}, prefix, temporal_options)
                        rows.append(row)
                        status = 'incomplete_after_inference_failure'
                        incomplete = True
                        break
                    try:
                        if frame.gt_error:
                            row['status'] = 'invalid_gt'
                            failures.append({**prefix,'index':index,'stage':'ground_truth','reason':frame.gt_error})
                        elif frame.gt is None:
                            row['status'] = 'unmatched'
                        else:
                            row['gt_sha256'] = array_digest(frame.gt)
                            row.update(score_depth(pred, frame.gt, frame.gt_valid_mask))
                            row['raw_delta1'] = row.get('delta1', '')
                            if row['status'] == 'ok' and cfg.get('compute_ssi', False):
                                row.update(score_ssi(pred, frame.gt, frame.gt_valid_mask))
                            if row['status'] == 'ok' and cfg.get('compute_upstream_depth_metrics', False):
                                from svr_depth.evaluation.upstream_depth_metrics import score_upstream_depth
                                if temporal_rows is not None:
                                    upstream, aligned = score_upstream_depth(frame.gt, pred, frame.gt_valid_mask, return_aligned=True)
                                else:
                                    upstream = score_upstream_depth(frame.gt, pred, frame.gt_valid_mask)
                                row.update(upstream)
                                row['upstream_delta1_ssi'] = upstream.get('upstream_depth_d1_ssi', '')
                            if row['status'] != 'ok':
                                failures.append({**prefix,'index':index,'stage':'depth_metric','reason':row['status']})
                        if cfg.get('save_predictions', False):
                            dest = out/'predictions'/_slug(spec['dataset'])/_slug(spec['sequence'])/_slug(method['method']+'_'+method['seed'])
                            dest.mkdir(parents=True, exist_ok=True)
                            payload = {'prediction_m':pred}
                            if frame.gt is not None:
                                payload.update(gt_m=frame.gt, gt_valid_mask=frame.gt_valid_mask)
                            np.savez_compressed(dest/f'{index:08d}.npz', **payload)
                    except Exception as exc:
                        row['status'] = 'metric_or_output_failure'
                        failures.append({**prefix,'index':index,'stage':'metric_or_output','reason':f'{type(exc).__name__}: {exc}'})
                        status = 'incomplete_after_metric_or_output_failure'
                        incomplete = True
                    current = {'frame':frame, 'prediction':pred, 'meta':meta, 'ssi_aligned':aligned}
                    if temporal_rows is not None and previous is not None:
                        _append_temporal(temporal_rows, previous, current, prefix, temporal_options)
                    previous = current
                    rows.append(row)
            except Exception as exc:
                failures.append({**prefix,'index':index,'stage':'source','reason':f'source failure: {type(exc).__name__}: {exc}'})
                status = 'incomplete_after_source_failure'
                incomplete = True
            calls.append({**prefix,'status':status, **engine.stats(), 'peak_cuda_allocated_bytes':engine.peak_bytes()})
            _write_progress(out, rows, calls, failures, temporal_rows)
        del engine
    _write_quality(out, rows, calls, failures, int(cfg.get('bootstrap_samples', 2000)),
                   complete=not incomplete, development=cfg.get('development_frame_limit') is not None,
                   temporal_rows=temporal_rows)
    return {'complete':not incomplete, 'checkpoint_metadata':identities,
            'failure_rows':len(failures), 'frame_rows':len(rows),
            'temporal_rows':len(temporal_rows) if temporal_rows is not None else 0,
            'temporal_scope':temporal_options.get('protocol', 'provisional_v1') if temporal_rows is not None else 'disabled',
            'runtime_environment':runtime}


def run_timing(cfg, out, engine_factory, base_sha, *, iterator_factory=iter_sequence):
    repeats = int(cfg.get('timing_repeats', 3))
    warmup = int(cfg.get('timing_warmup_frames', 16))
    if repeats < 3 or warmup < 0:
        raise ValueError('Timing requires >=3 repeats and nonnegative warmup frames')
    all_times, calls, failures = [], [], []
    runtime = None
    methods = cfg['methods']
    for repeat in range(repeats):
        rotated = methods[repeat % len(methods):] + methods[:repeat % len(methods)]
        for method in rotated:
            engine = engine_factory(cfg['snapshot']['path'], method, base_sha)
            if runtime is None and hasattr(engine, 'runtime_info'):
                runtime = engine.runtime_info()
            for spec in cfg['sequences']:
                engine.reset()
                prefix = {'dataset':spec['dataset'],'scene':spec['scene'],'sequence':spec['sequence'],
                          'method':method['method'],'seed':method['seed'],'repeat':repeat+1}
                print(f'[M01 timing] {prefix}', flush=True)
                count, status = 0, 'complete'
                try:
                    frames = iterator_factory(spec, load_gt=False)
                    if cfg.get('development_frame_limit') is not None:
                        frames = itertools.islice(frames, cfg['development_frame_limit'])
                    iterator = iter(frames)
                    while True:
                        engine.synchronize()
                        start = time.perf_counter()
                        try:
                            frame = next(iterator)
                        except StopIteration:
                            break
                        decoded = time.perf_counter()
                        prediction, meta = engine.step(frame.index, frame.rgb)
                        # Include device-to-host transfer just as in quality output.
                        engine.to_numpy(prediction)
                        engine.synchronize()
                        end = time.perf_counter()
                        if frame.index >= warmup:
                            all_times.append({**prefix, 'index':frame.index, 'age':meta['age'],
                                              'keyframe':meta['keyframe'], 'latency_ms':(end-start)*1000.0,
                                              'source_decode_ms':(decoded-start)*1000.0,
                                              'engine_d2h_ms':(end-decoded)*1000.0})
                            count += 1
                    if count == 0:
                        raise ValueError(f'no measured frames after {warmup} warmup frames')
                except Exception as exc:
                    failures.append({**prefix, 'index':'', 'reason':f'{type(exc).__name__}: {exc}'})
                    status = 'incomplete'
                calls.append({**prefix,'status':status, **engine.stats(), 'peak_cuda_allocated_bytes':engine.peak_bytes()})
                _csv(out/'timing_frames.csv', all_times, ['dataset','scene','sequence','method','seed','repeat','index','latency_ms'])
                latency = latency_summary(all_times)
                _csv(out/'latency_summary.csv', latency, ['dataset','scene','sequence','method','seed','repeat'])
                _csv(out/'latency_dataset_summary.csv', latency_hierarchy(latency), ['dataset','method','seed'])
                _csv(out/'call_counts.csv', calls, ['dataset','scene','sequence','method','seed','repeat','status'])
                _csv(out/'failures.csv', failures, ['dataset','scene','sequence','method','seed','repeat','reason'])
            del engine
    return {'complete':not failures, 'failure_rows':len(failures), 'timed_frame_rows':len(all_times), 'runtime_environment':runtime}


def execute(config_path, out, mode, engine_factory=None, development_frames=None):
    cfg = load_config(config_path)
    if development_frames is not None:
        if development_frames < 1:
            raise ValueError('--development-frames must be positive')
        if mode == 'timing' and development_frames <= int(cfg.get('timing_warmup_frames', 16)):
            raise ValueError('development frame limit must exceed timing warmup')
        cfg['development_frame_limit'] = int(development_frames)
    identity = _source_identity(cfg, config_path, mode)
    identity['run_scope'] = 'development_only_truncated' if development_frames is not None else 'configured_full_sequences'
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    _json(out/'run_identity.json', {**identity, 'status':'running'})
    if engine_factory is None:
        from svr_depth.streaming.m01_engine import M01Engine
        engine_factory = M01Engine
    try:
        if mode == 'quality':
            result = run_quality(cfg, out, engine_factory, identity['backbone']['weights_sha256'])
        elif mode == 'timing':
            result = run_timing(cfg, out, engine_factory, identity['backbone']['weights_sha256'])
        else:
            raise ValueError(f'Unknown mode {mode}')
        identity.update(result, status='complete' if result['complete'] else 'incomplete')
        _json(out/'run_identity.json', identity)
        return result
    except Exception as exc:
        identity.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        _json(out/'run_identity.json', identity)
        raise
