"""M01 descriptive, paired-risk and cluster/seed bootstrap summaries."""
from __future__ import annotations
from collections import defaultdict
import math
import random
import statistics


def _mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def _quantile(values, q):
    a = sorted(values)
    if not a:
        return None
    i = (len(a)-1)*q
    lo, hi = int(i), min(int(i)+1, len(a)-1)
    return a[lo] + (a[hi]-a[lo])*(i-lo)


def _group(rows, names):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in names)].append(r)
    return groups


def _number(x):
    if x is None or x == '':
        return None
    y = float(x)
    return y if math.isfinite(y) else None


def _ok(r):
    return r['status'] == 'ok' and _number(r.get('absrel')) is not None


def summarize_frames(rows, bootstrap_samples=2000, bootstrap_seed=20261106):
    keys = ('dataset', 'scene', 'sequence', 'method', 'seed')
    seen = set()
    for row in rows:
        key = tuple(row[x] for x in keys) + (int(row['index']),)
        if key in seen:
            raise ValueError(f'duplicate frame identity {key}')
        seen.add(key)
    sequences = []
    for key, group in sorted(_group(rows, keys).items()):
        valid = [r for r in group if _ok(r)]
        eligible = [r for r in group if r['status'] != 'unmatched']
        failed = [r for r in eligible if not _ok(r)]
        row = dict(zip(keys, key))
        row.update(total_frames=len(group), eligible_gt_frames=len(eligible), scored_frames=len(valid),
                   failed_gt_frames=len(failed), failure_rate=len(failed)/len(eligible) if eligible else None,
                   absrel=_mean(float(r['absrel']) for r in valid),
                   rmse=_mean(float(r['rmse']) for r in valid),
                   delta1=_mean(float(r['delta1']) for r in valid),
                   frame_absrel_p95=_quantile([float(r['absrel']) for r in valid], .95),
                   ssi_absrel=_mean(float(r['ssi_absrel']) for r in valid if _number(r.get('ssi_absrel')) is not None),
                   ssi_rmse=_mean(float(r['ssi_rmse']) for r in valid if _number(r.get('ssi_rmse')) is not None),
                   ssi_delta1=_mean(float(r['ssi_delta1']) for r in valid if _number(r.get('ssi_delta1')) is not None),
                   upstream_depth_d1=_mean(float(r['upstream_depth_d1']) for r in valid if r.get('upstream_depth_status') == 'ok' and _number(r.get('upstream_depth_d1')) is not None),
                   upstream_depth_d1_ssi=_mean(float(r['upstream_depth_d1_ssi']) for r in valid if r.get('upstream_depth_status') == 'ok' and _number(r.get('upstream_depth_d1_ssi')) is not None),
                   upstream_ssi_nonpositive_fraction=_mean(float(r['upstream_ssi_nonpositive_fraction']) for r in valid if r.get('upstream_depth_status') == 'ok' and _number(r.get('upstream_ssi_nonpositive_fraction')) is not None))
        sequences.append(row)
    scenes = []
    scene_keys = ('dataset', 'scene', 'method', 'seed')
    for key, group in sorted(_group(sequences, scene_keys).items()):
        row = dict(zip(scene_keys, key))
        row.update(sequences=len(group), scored_sequences=sum(r['scored_frames'] > 0 for r in group),
                   absrel=_mean(r['absrel'] for r in group if r['absrel'] is not None),
                   rmse=_mean(r['rmse'] for r in group if r['rmse'] is not None),
                   delta1=_mean(r['delta1'] for r in group if r['delta1'] is not None),
                   frame_absrel_p95=_mean(r['frame_absrel_p95'] for r in group if r['frame_absrel_p95'] is not None),
                   failure_rate=_mean(r['failure_rate'] for r in group if r['failure_rate'] is not None),
                   **{k:_mean(r[k] for r in group if r[k] is not None) for k in ('ssi_absrel','ssi_rmse','ssi_delta1','upstream_depth_d1','upstream_depth_d1_ssi','upstream_ssi_nonpositive_fraction')})
        scenes.append(row)
    dataset = []
    rng = random.Random(bootstrap_seed)
    for (ds, method), group in sorted(_group(scenes, ('dataset', 'method')).items()):
        by_seed = _group(group, ('seed',))
        scene_sets = [{r['scene'] for r in rr if r['absrel'] is not None} for rr in by_seed.values()]
        common = set.intersection(*scene_sets) if scene_sets else set()
        same_coverage = bool(common) and all(s == common for s in scene_sets)
        complete = same_coverage and all(r['absrel'] is not None for r in group)
        per_seed = []
        for (seed,), sr in sorted(by_seed.items()):
            per_seed.append({'seed': seed, 'scenes': len(sr),
                             **{k: _mean(r[k] for r in sr if r[k] is not None)
                                for k in ('absrel', 'rmse', 'delta1', 'frame_absrel_p95', 'failure_rate', 'ssi_absrel', 'ssi_rmse', 'ssi_delta1', 'upstream_depth_d1', 'upstream_depth_d1_ssi', 'upstream_ssi_nonpositive_fraction')}})
        interval = None
        if complete and len(common) >= 2 and bootstrap_samples > 0:
            lookup = {(r['seed'], r['scene']): r['absrel'] for r in group}
            seeds = sorted(k[0] for k in by_seed)
            scene_names = sorted(common)
            boot = []
            for _ in range(bootstrap_samples):
                draw_seeds = [rng.choice(seeds) for _ in seeds]
                draw_scenes = [rng.choice(scene_names) for _ in scene_names]
                boot.append(_mean(lookup[s, c] for s in draw_seeds for c in draw_scenes))
            interval = [_quantile(boot, .025), _quantile(boot, .975)]
        dataset.append({'dataset': ds, 'method': method, 'per_seed': per_seed,
                        'across_seed_mean_absrel': _mean(r['absrel'] for r in per_seed if r['absrel'] is not None),
                        'absrel_ci95': interval, 'ci_method': 'joint resampling of scene groups and training seeds',
                        'ci_available': interval is not None, 'consistent_scene_coverage': same_coverage,
                        'confidence_limitation': 'Uncertainty over seeds cannot be estimated with one seed; intervals require >=2 scenes.'})
    return sequences, scenes, dataset


def paired_risk(rows):
    native = {}
    by_method = {}
    for r in rows:
        key = (r['dataset'], r['sequence'], int(r['index']))
        if r['method'] == 'A':
            if key in native:
                raise ValueError(f'A must have exactly one native record per frame: {key}')
            native[key] = r
        by_method[(r['dataset'], r['sequence'], int(r['index']), r['method'], r['seed'])] = r
    result = []
    candidates = [r for r in rows if r['method'] != 'A']
    for key, group in sorted(_group(candidates, ('dataset','scene','sequence','method','seed')).items()):
        references = ['A']
        if key[3] == 'P_multi' and any(
            k[0] == key[0] and k[1] == key[2] and k[3] == 'O_fast' and k[4] == key[4]
            for k in by_method
        ):
            references.append('O_fast')
        for reference in references:
            ds, scene, seq, method, seed = key
            diffs, missing = [], 0
            for r in group:
                idx = int(r['index'])
                ref = native.get((ds, seq, idx)) if reference == 'A' else by_method.get((ds, seq, idx, reference, seed))
                if ref is not None and ref.get('rgb_sha256') and r.get('rgb_sha256') and ref['rgb_sha256'] != r['rgb_sha256']:
                    raise ValueError(f'Paired rgb_sha256 mismatch: {ds}/{seq}/{idx}, {method} vs {reference}')
                if ref is None or not _ok(ref) or not _ok(r):
                    missing += 1
                    continue
                for field in ('mask_sha256', 'rgb_sha256', 'gt_sha256'):
                    if not r.get(field) or r.get(field) != ref.get(field):
                        raise ValueError(f'Paired {field} mismatch: {ds}/{seq}/{idx}, {method} vs {reference}')
                diffs.append(float(r['absrel']) - float(ref['absrel']))
            n = len(diffs)
            result.append({'dataset': ds, 'scene': scene, 'sequence': seq, 'method': method, 'seed': seed,
                           'reference': reference, 'total_frames': len(group), 'paired_valid_frames': n,
                           'unpaired_or_invalid_frames': missing, 'mean_absrel_delta': _mean(diffs),
                           'harmed_fraction': sum(d > 0 for d in diffs)/n if n else None,
                           'severe_fraction': sum(d > .01 for d in diffs)/n if n else None,
                           'severe_threshold': .01, 'denominator': 'successfully scored common GT frames only'})
    return result


def latency_summary(rows):
    out = []
    for key, rr in sorted(_group(rows, ('dataset','scene','sequence','method','seed','repeat')).items()):
        times = [float(r['latency_ms']) for r in rr if r.get('latency_ms') not in ('', None)]
        if not times:
            continue
        mean = _mean(times)
        out.append({**dict(zip(('dataset','scene','sequence','method','seed','repeat'), key)),
                    'timed_frames': len(times), 'mean_ms': mean, 'p50_ms': _quantile(times,.5),
                    'p95_ms': _quantile(times,.95), 'fps': 1000/mean,
                    'mean_source_decode_ms': _mean(float(r['source_decode_ms']) for r in rr if r.get('source_decode_ms') not in ('', None)),
                    'mean_engine_d2h_ms': _mean(float(r['engine_d2h_ms']) for r in rr if r.get('engine_d2h_ms') not in ('', None)),
                    'scope': 'RGB read/decode + preprocess + DIS + inference + D2H + synchronization; excludes GT, scoring, prediction storage and model loading'})
    return out


def risk_hierarchy(risk, bootstrap_samples=2000, bootstrap_seed=20261106):
    """Paired frame deltas -> sequence -> physical scene -> dataset/seed.

    Each sequence's paired quantities are averaged equally within a scene,
    and each physical scene equally within a dataset. Bootstrap keeps the
    paired method/reference delta intact and resamples each seed as a whole.
    """
    fields = ('mean_absrel_delta','harmed_fraction','severe_fraction')
    scenes = []
    for key, group in sorted(_group(risk, ('dataset','scene','method','seed','reference')).items()):
        valid = [r for r in group if r['mean_absrel_delta'] is not None]
        scenes.append({**dict(zip(('dataset','scene','method','seed','reference'),key)),
                       'sequences':len(group), 'sequences_with_pairs':len(valid),
                       'paired_valid_frames':sum(r['paired_valid_frames'] for r in group),
                       'unpaired_or_invalid_frames':sum(r['unpaired_or_invalid_frames'] for r in group),
                       **{f:_mean(r[f] for r in valid) for f in fields}})
    datasets = []
    for key, group in sorted(_group(scenes, ('dataset','method','seed','reference')).items()):
        valid = [r for r in group if r['mean_absrel_delta'] is not None]
        datasets.append({**dict(zip(('dataset','method','seed','reference'),key)),
                         'scenes':len(group), 'scenes_with_pairs':len(valid),
                         'paired_valid_frames':sum(r['paired_valid_frames'] for r in group),
                         'unpaired_or_invalid_frames':sum(r['unpaired_or_invalid_frames'] for r in group),
                         **{f:_mean(r[f] for r in valid) for f in fields},
                         'aggregation':'equal sequence weight within scene, equal scene weight within dataset'})
    combined = []
    rng = random.Random(bootstrap_seed)
    for key, group in sorted(_group(scenes, ('dataset','method','reference')).items()):
        by_seed = _group(group, ('seed',))
        coverage = [{r['scene'] for r in rr if r['mean_absrel_delta'] is not None} for rr in by_seed.values()]
        common = set.intersection(*coverage) if coverage else set()
        full = bool(common) and all(v == common for v in coverage) and all(r['mean_absrel_delta'] is not None for r in group)
        point = _mean(r['mean_absrel_delta'] for r in datasets
                      if (r['dataset'],r['method'],r['reference']) == key and r['mean_absrel_delta'] is not None)
        interval = None
        if full and len(common) >= 2 and bootstrap_samples > 0:
            lookup = {(r['seed'],r['scene']):r['mean_absrel_delta'] for r in group}
            seeds, scene_names = sorted(x[0] for x in by_seed), sorted(common)
            values=[]
            for _ in range(bootstrap_samples):
                sampled_seeds=[rng.choice(seeds) for _ in seeds]
                sampled_scenes=[rng.choice(scene_names) for _ in scene_names]
                values.append(_mean(lookup[s,c] for s in sampled_seeds for c in sampled_scenes))
            interval=[_quantile(values,.025),_quantile(values,.975)]
        combined.append({**dict(zip(('dataset','method','reference'),key)),
                         'training_seed_count':len(by_seed), 'common_scene_count':len(common),
                         'consistent_scene_coverage':full, 'mean_absrel_delta':point,
                         'paired_absrel_delta_ci95':interval,
                         'ci_method':'paired scene-group and whole-seed bootstrap; unavailable with <2 scenes',
                         'limitation':'one seed cannot estimate seed uncertainty'})
    return scenes, datasets, combined


def latency_hierarchy(summary):
    scene_rows=[]
    for key, group in sorted(_group(summary, ('dataset','scene','method','seed','repeat')).items()):
        scene_rows.append({**dict(zip(('dataset','scene','method','seed','repeat'),key)),
                           'sequences':len(group), 'mean_ms':_mean(r['mean_ms'] for r in group)})
    trial_rows=[]
    for key, group in sorted(_group(scene_rows, ('dataset','method','seed','repeat')).items()):
        trial_rows.append({**dict(zip(('dataset','method','seed','repeat'),key)),
                           'scenes':len(group), 'mean_ms':_mean(r['mean_ms'] for r in group)})
    result=[]
    for key, group in sorted(_group(trial_rows, ('dataset','method','seed')).items()):
        means=[r['mean_ms'] for r in group]
        result.append({**dict(zip(('dataset','method','seed'),key)), 'repeats':len(means),
                       'repeat_mean_latency_ms':_mean(means), 'repeat_median_latency_ms':_quantile(means,.5),
                       'scene_counts_by_repeat':','.join(str(r['scenes']) for r in group),
                       'aggregation':'equal sequence weight within scene, equal scene weight within dataset'})
    return result
