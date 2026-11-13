"""Descriptive paired risk against a per-frame native baseline."""
from collections import defaultdict

def summarize_against_baseline(rows, baseline='A', severe_delta=0.01, baseline_seed=None):
    if severe_delta < 0:
        raise ValueError('severe_delta must be nonnegative')
    native = {}
    for r in rows:
        if r['method'] == baseline and (baseline_seed is None or r['seed'] == baseline_seed):
            k = (r['dataset'], r['sequence'], r['k'], r['index'])
            if k in native:
                raise ValueError(f'Ambiguous baseline for {k}')
            native[k] = r
    if not native:
        raise ValueError(f'No baseline records for {baseline!r}')
    groups = defaultdict(list)
    for r in rows:
        if r['method'] != baseline:
            groups[(r['dataset'], r['sequence'], r['method'], r['seed'], r['k'])].append(r)
    results = []
    for key, items in sorted(groups.items()):
        candidate_indexes = {r['index'] for r in items}
        baseline_indexes = {index for dataset, sequence, k, index in native
                            if (dataset, sequence, k) == (key[0], key[1], key[4])}
        if candidate_indexes != baseline_indexes:
            raise ValueError(f'Incomplete candidate or baseline frame coverage: {key}')
        deltas, missing = [], 0
        for r in items:
            a = native.get((r['dataset'], r['sequence'], r['k'], r['index']))
            if a is None or r['status'] != 'ok' or a['status'] != 'ok':
                missing += 1
                continue
            if r['mask_sha256'] != a['mask_sha256']:
                raise ValueError(f'Mismatched GT masks: {r["dataset"]}/{r["sequence"]}/{r["index"]}')
            deltas.append(r['absrel'] - a['absrel'])
        n = len(deltas)
        results.append(dict(dataset=key[0], sequence=key[1], method=key[2], seed=key[3], k=key[4],
            total_frames=len(items), paired_valid=n, missing_or_invalid=missing,
            mean_absrel_delta=sum(deltas)/n if n else None,
            harmed_fraction=sum(d > 0 for d in deltas)/n if n else None,
            severe_fraction=sum(d > severe_delta for d in deltas)/n if n else None,
            severe_delta=severe_delta, interpretation='conditional_on_valid_pairs'))
    return results
