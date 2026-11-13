"""Locked development protocol and CPU analysis for Y08-Cal / Y09-Pareto."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from collections import defaultdict

VERSION = 'Y08CAL-Y09P-1.0'
SEQS = ('crowd2', 'crowd3', 'balloon2')
TAIL = dict(crowd2=768, crowd3=256, balloon2=256)
FAMILIES = ('B2', 'O_multi', 'O_fast', 'P')
SEED = 20261025
STEPS = 1000
SLOTS = [(s, c) for s in SEQS for c in range(8)]
FLOOR = 1e-4
HARM_TOL = 1e-8


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def json_write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    t.replace(p)


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_write(path, rows):
    require(bool(rows), f'Empty output table: {path}')
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with t.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    t.replace(p)


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def training_order():
    rng = random.Random(SEED); result = []
    while len(result) < STEPS:
        block = list(range(24)); rng.shuffle(block); result.extend(block)
    return result[:STEPS]


def validate_recovery(curve, step, order, exposure):
    require(0 < step <= len(order), 'Recovery step outside locked order')
    require([r['step'] for r in curve]==list(range(1,step+1)), 'Duplicate/missing recovered steps')
    require(all(r['slot']==order[r['step']-1] for r in curve), 'Recovery slot order changed')
    require(sum(r['supervised_frames'] for r in curve)==exposure, 'Recovery exposure count changed')


def config(family, schedule='K32', gain=1., frequency=1):
    require(family in ('A', *FAMILIES), 'Unknown method')
    require(schedule in ('K16', 'K32', 'K64', 'Auto'), 'Unknown schedule')
    require(gain in (0., .25, .5, 1.) and frequency in (1, 4), 'Unregistered gain/frequency')
    if gain == 0 or family == 'A':
        family, gain, frequency = 'A', 0., 1
    key = f'{family}_{schedule}_g{gain:g}_f{frequency}'
    return dict(id=key, family=family, schedule=schedule, gain=gain, frequency=frequency)


def plan():
    # Eight physical settings per learned family, without a Cartesian explosion.
    settings = [('K32', .25, 1), ('K32', .5, 1), ('K32', 1., 1),
                ('K16', 1., 1), ('K64', 1., 1), ('Auto', 1., 1),
                ('K32', 1., 4), ('K64', 1., 4)]
    result = [config('A', s) for s in ('K16', 'K32', 'K64', 'Auto')]
    result += [config(f, *setting) for f in FAMILIES for setting in settings]
    require(len(result) == 36 and len({r['id'] for r in result}) == 36, 'Plan has duplicates')
    return result


def repair_due(age, frequency):
    return age > 0 and (age - 1) % frequency == 0


def is_fit_frame(index, matched=True):
    # Shared *frame exposure*, not a method-dependent keyframe/propagation filter.
    return matched and 0 < index < 256 and index % 32 != 0


def keep_prediction(seq, index):
    return index < 256 or index >= TAIL[seq]


class Moments:
    """Stable global OLS, fitted to training pixels only; no scene-specific fit."""
    def __init__(self):
        self.n = 0; self.x = self.y = self.xx = self.xy = 0.

    def add(self, pred, gt, np):
        m = (gt > 0) & np.isfinite(gt)
        require(np.all(np.isfinite(pred[m]) & (pred[m] > 0)), 'Invalid training prediction')
        x, y = pred[m].astype('float64'), gt[m].astype('float64')
        if not x.size: return
        n = x.size; mx = float(x.mean()); my = float(y.mean())
        xx = float(((x-mx)**2).sum()); xy = float(((x-mx)*(y-my)).sum())
        if self.n:
            N = self.n+n; dx = mx-self.x; dy = my-self.y; weight = self.n*n/N
            self.xx += xx + dx*dx*weight; self.xy += xy + dx*dy*weight
            self.x += dx*n/N; self.y += dy*n/N; self.n = N
        else:
            self.n, self.x, self.y, self.xx, self.xy = n, mx, my, xx, xy

    def finish(self):
        require(self.n >= 1000 and self.xx > 1e-12, 'Ill-conditioned calibration fit')
        a = self.xy/self.xx; b = self.y-a*self.x
        require(math.isfinite(a) and math.isfinite(b) and a > 0, 'Invalid scale/bias')
        return dict(scale=a, bias_m=b, floor_m=FLOOR, fit_pixels=self.n,
                    fit_scope='shared matched propagation RGB indices 1..255, index%32!=0; all three training prefixes',
                    method='global pixel OLS; positive output floor fixed before evaluation',
                    moments=dict(n=self.n, mean_x=self.x, mean_y=self.y, centered_xx=self.xx, centered_xy=self.xy))

    def merge(self, other):
        if not other.n: return
        if not self.n:
            self.__dict__.update(other.__dict__);return
        N=self.n+other.n;dx=other.x-self.x;dy=other.y-self.y;w=self.n*other.n/N
        self.xx+=other.xx+dx*dx*w;self.xy+=other.xy+dx*dy*w
        self.x+=dx*other.n/N;self.y+=dy*other.n/N;self.n=N


def metrics(pred, gt, np):
    mask = (gt > 0) & np.isfinite(gt)
    require(mask.any(), 'No valid GT pixels')
    require(np.all(np.isfinite(pred[mask]) & (pred[mask] > 0)), 'Invalid prediction on GT-valid pixels')
    p, g = pred[mask].astype('float64'), gt[mask].astype('float64')
    return dict(AbsRel=float((abs(p-g)/g).mean()), RMSE=float(((p-g)**2).mean()**.5),
                delta1=float((np.maximum(p/g, g/p) < 1.25).mean()), valid_pixels=int(mask.sum()))


def quantile(values, q):
    values = sorted(values); require(bool(values), 'Empty quantile')
    k = (len(values)-1)*q; lo = int(k); hi = min(lo+1, len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(k-lo)


def quality_summary(rows):
    """All families use common index/32 windows, including Auto and K64."""
    groups = defaultdict(list)
    for r in rows:
        if r['scored'] and r['matched']:
            groups[r['variant'], r['sequence']].append(r)
    base = {(r['sequence'], r['index']): r for r in rows
            if r['variant'] == 'A_K32_g0_f1/raw' and r['scored'] and r['matched']}
    result = []
    for (variant, seq), rr in sorted(groups.items()):
        windows = defaultdict(list)
        for r in rr: windows[r['index']//32].append(r)
        whole = {r['index'] for r in rows if r['variant']==variant and r['sequence']==seq and r['scored']}
        full = [ww for w,ww in windows.items() if set(range(w*32, w*32+32)).issubset(whole)]
        harm = sum(r['AbsRel'] > base[seq,r['index']]['AbsRel']+HARM_TOL for r in rr)
        hc = sum(statistics.mean(r['AbsRel']-base[seq,r['index']]['AbsRel'] for r in ww)>HARM_TOL for ww in full)
        result.append(dict(variant=variant, sequence=seq, scored_GT_frames=len(rr),
            mean_AbsRel=statistics.mean(r['AbsRel'] for r in rr), mean_RMSE=statistics.mean(r['RMSE'] for r in rr),
            mean_delta1=statistics.mean(r['delta1'] for r in rr), frame_AbsRel_p95=quantile([r['AbsRel'] for r in rr], .95),
            harmed_frames_vs_raw_A_K32=harm, harmed_fraction=harm/len(rr),
            complete_common_32RGB_windows=len(full), harmed_common_windows=hc,
            risk_reference='same GT frames / raw A K32; never method-dependent propagation subset'))
    return result


def dominates(a, b):
    axes = ('AbsRel','RMSE','worst_sequence_AbsRel','harm_fraction','frame_AbsRel_p95','latency_ms')
    return (all(a[k] <= b[k] for k in axes) and a['delta1'] >= b['delta1'] and
            (any(a[k] < b[k] for k in axes) or a['delta1'] > b['delta1']))


def pareto_rows(quality, timing):
    qq = defaultdict(list); tt = defaultdict(list)
    for r in quality: qq[r['variant']].append(r)
    for r in timing: tt[r['variant'],r['sequence']].append(r)
    candidates = []
    for v, rr in sorted(qq.items()):
        require(len(rr)==3, 'Macro requires all three sequences')
        if not all((v,s) in tt for s in SEQS): continue
        trial = [x for s in SEQS for x in tt[v,s]]
        if not all(x['quality_checked'] and x['endpoint_clean'] for x in trial): continue
        candidates.append(dict(variant=v, AbsRel=statistics.mean(r['mean_AbsRel'] for r in rr),
            RMSE=statistics.mean(r['mean_RMSE'] for r in rr), delta1=statistics.mean(r['mean_delta1'] for r in rr),
            harm_fraction=statistics.mean(r['harmed_fraction'] for r in rr),
            frame_AbsRel_p95=statistics.mean(r['frame_AbsRel_p95'] for r in rr),
            latency_ms=statistics.mean(statistics.median(x['tail_mean_ms'] for x in tt[v,s]) for s in SEQS),
            worst_sequence_AbsRel=max(r['mean_AbsRel'] for r in rr),
            scope='three development videos; one training seed; strict measured-point dominance only'))
    for r in candidates:
        r['dominated_by'] = ';'.join(x['variant'] for x in candidates if dominates(x,r))
        r['nondominated'] = not r['dominated_by']
    return candidates
