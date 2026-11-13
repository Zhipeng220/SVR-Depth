#!/usr/bin/env python3
"""M01 manifest freeze and strict paired-results aggregation; Python stdlib only."""
import argparse, csv, hashlib, json, math, random, statistics
from collections import defaultdict
from pathlib import Path

METHODS = ('A', 'O_fast', 'O_multi', 'P')
METRICS = ('AbsRel', 'RMSE', 'delta1')

def check(ok, msg):
    if not ok: raise ValueError(msg)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def read(p):
    with open(p, encoding='utf-8-sig') as f: return json.load(f)

def rows(p):
    with open(p, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def write_json(p, obj):
    p = Path(p); check(not p.exists(), f'Output already exists: {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8'); tmp.replace(p)

def write_csv(p, data):
    check(data, f'No data for {p}')
    p = Path(p); check(not p.exists(), f'Output already exists: {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

def freeze(config_path, output):
    c = read(config_path); base = Path(config_path).resolve().parent
    check(c.get('protocol') == 'M01-v1', 'Wrong protocol')
    check(c.get('test_unsealed') is False, 'Test already unsealed or audit status unspecified')
    check(c.get('k') == 16 and c.get('gain') == 1 and c.get('frequency') == 1 and c.get('alignment') == 'raw', 'M01 working point must be K16/g1/f1/raw')
    check(c.get('depth_contract_confirmed') is True, 'Confirm metric z-depth, scale, registration and mask contract after M00 audit')
    check(c.get('independence_audited') is True, 'Complete independence audit before freezing')
    seqs = c.get('sequences', []); check(seqs, 'No sequences')
    check(len({s['sequence'] for s in seqs}) == len(seqs), 'Sequence identifiers must be globally unique')
    check(all(s.get('dataset') and s.get('physical_scene_group') and s.get('camera_setup') and s.get('sensor_model') and s.get('recording_session') and s.get('independence_notes') and s.get('depth_scale_to_meters', 0) > 0 for s in seqs), 'Incomplete sequence audit')
    check(len({s['dataset'] for s in seqs}) == 2 and any(s['dataset'].lower() == 'bonn' for s in seqs), 'Specify Bonn and exactly one second source')
    check(set(c['checkpoints']) == set(METHODS[1:]), 'Supply checkpoints for O_fast, O_multi, P')
    ids = {m: [str(x['seed']) for x in c['checkpoints'][m]] for m in METHODS[1:]}
    check(all(len(v) == 3 and len(set(v)) == 3 for v in ids.values()), 'Exactly three distinct seeds per trained method required')
    files = {}
    def add(name):
        p = Path(name); p = p if p.is_absolute() else base / p; p = p.resolve()
        check(p.is_file(), f'Missing input: {p}'); files[str(p)] = sha(p); return str(p)
    c['model_files'] = [add(p) for p in c['model_files']]; check(c['model_files'], 'Model files required')
    c['source_files'] = [add(p) for p in c['source_files']]; check(c['source_files'], 'Source files required')
    for m in METHODS[1:]:
        for x in c['checkpoints'][m]: x['path'] = add(x['path'])
    for s in seqs:
        s['manifest'] = add(s['manifest']); rr = rows(s['manifest']); check(rr, 'Empty manifest')
        check([int(r['frame_index_zero_based']) for r in rr] == list(range(len(rr))), 'Manifest indices must be continuous from zero')
        check(any(r['gt_match_status'].upper() == 'MATCHED' for r in rr), 'No matched GT')
        for r in rr:
            check(r['gt_match_status'].upper() in ('MATCHED', 'UNMATCHED'), 'Invalid GT status')
            add(r['rgb_path'])
            if r['gt_match_status'].upper() == 'MATCHED': add(r['depth_path'])
        s['rgb_count'] = len(rr); s['matched_count'] = sum(r['gt_match_status'].upper() == 'MATCHED' for r in rr)
    c['input_sha256'] = files
    c['freeze_warning'] = 'Attestations are operator supplied; hashing does not prove independence or correct calibration.'
    write_json(output, c)
    print(f'Frozen {len(seqs)} sequences and {len(files)} input files: {output}')

def percentile(v, q):
    a = sorted(v); x = (len(a)-1)*q; lo = int(x); hi = min(lo+1, len(a)-1); return a[lo] + (a[hi]-a[lo])*(x-lo)

def aggregate(protocol, frames_path, timing_path, out):
    c = read(protocol)
    for p,h in c['input_sha256'].items(): check(Path(p).is_file() and sha(p) == h, f'Frozen input changed: {p}')
    fs = rows(frames_path); ts = rows(timing_path); seqs = {s['sequence']:s for s in c['sequences']}
    expected = [('A','native')] + [(m,str(x['seed'])) for m in METHODS[1:] for x in c['checkpoints'][m]]
    data = defaultdict(dict)
    for r in fs:
        key = (r['sequence'],r['method'],r['seed']); i = int(r['index'])
        check(key[0] in seqs and (key[1],key[2]) in expected and i not in data[key], f'Unexpected/duplicate frame {key} {i}')
        data[key][i] = r
    for s, meta in seqs.items():
        for m, seed in expected:
            check(set(data[s,m,seed]) == set(range(meta['rgb_count'])), f'Incomplete RGB trajectory: {s}/{m}/{seed}')
    time_map = defaultdict(list)
    for r in ts:
        key=(r['sequence'],r['method'],r['seed']); check(key[0] in seqs and (key[1],key[2]) in expected,'Unexpected timing unit')
        check(r['verified'].lower() == 'true', f'Unverified timing: {key}')
        val=float(r['mean_ms']); check(math.isfinite(val) and val > 0, 'Invalid timing'); time_map[key].append((r['repeat'],val))
    for s in seqs:
        for m,seed in expected:
            t=time_map[s,m,seed]; check(len(t)==3 and len({r for r,_ in t})==3, f'Need 3 verified timing repeats: {s}/{m}/{seed}')
    video=[]
    for s, meta in seqs.items():
        manifest=rows(meta['manifest']); a=data[s,'A','native']
        for m,seed in expected:
            rr=data[s,m,seed]; scored=[]; failed=0
            for i, src in enumerate(manifest):
                r=rr[i]; ref=a[i]; matched=src['gt_match_status'].upper()=='MATCHED'
                check(r['status'] in ('ok','unmatched','invalid_gt','invalid_prediction'), 'Unknown frame status')
                check((r['status']!='unmatched') == matched, f'GT pairing mismatch {s}/{i}')
                check((r['status']=='invalid_gt') == (ref['status']=='invalid_gt'), f'GT coverage differs {s}/{i}')
                if r['status'] in ('ok','invalid_prediction'):
                    check(r['mask_sha256'] and r['mask_sha256']==ref['mask_sha256'] and int(r['valid_pixels'])>0 and r['valid_pixels']==ref['valid_pixels'], f'GT mask differs {s}/{i}')
                if r['status']=='invalid_prediction' or ref['status']=='invalid_prediction': failed+=1; continue
                if r['status'] != 'ok': continue
                vals=[float(r[x]) for x in METRICS]; check(all(math.isfinite(x) for x in vals) and vals[0]>=0 and vals[1]>=0 and 0<=vals[2]<=1, 'Invalid metric')
                scored.append((r,ref))
            check(scored, f'No valid paired frames {s}/{m}/{seed}')
            delta=[float(r['AbsRel'])-float(a0['AbsRel']) for r,a0 in scored]
            v=dict(dataset=meta['dataset'],scene=meta['physical_scene_group'],sequence=s,method=m,seed=seed,RGB=len(rr),matched_GT=meta['matched_count'],valid_paired=len(scored),failed_paired=failed,
                   **{k:statistics.mean(float(r[k]) for r,_ in scored) for k in METRICS},p95_AbsRel=percentile([float(r['AbsRel']) for r,_ in scored],.95),harmed_fraction=sum(x>0 for x in delta)/len(delta),severe_fraction=sum(x>.01 for x in delta)/len(delta),latency_ms=statistics.median(v for _,v in time_map[s,m,seed]))
            video.append(v)
    # Strictly paired frames across every method: do not hide model-specific failures in comparative estimates.
    comparison=[]
    for s in seqs:
        for p in c['checkpoints']['P']:
            ps=str(p['seed'])
            for o in c['checkpoints']['O_fast']:
                os_=str(o['seed']); dif=[]
                for i in range(seqs[s]['rgb_count']):
                    a=data[s,'P',ps][i]; b=data[s,'O_fast',os_][i]
                    if a['status']=='ok' and b['status']=='ok': dif.append(float(a['AbsRel'])-float(b['AbsRel']))
                check(dif, 'No direct paired P/O_fast frames')
                comparison.append(dict(dataset=seqs[s]['dataset'],scene=seqs[s]['physical_scene_group'],sequence=s,P_seed=ps,O_fast_seed=os_,paired_frames=len(dif),P_minus_Ofast_AbsRel=statistics.mean(dif),P_minus_Ofast_gt_0_01=sum(x>.01 for x in dif)/len(dif)))
    groups=defaultdict(list)
    for r in video: groups[(r['dataset'],r['scene'],r['method'],r['seed'])].append(r)
    scene=[]
    keys=(*METRICS,'p95_AbsRel','harmed_fraction','severe_fraction','latency_ms')
    for (d,g,m,seed),v in sorted(groups.items()): scene.append(dict(dataset=d,scene=g,method=m,seed=seed,recordings=len(v),**{k:statistics.mean(x[k] for x in v) for k in keys}))
    macros=defaultdict(list)
    for r in scene: macros[(r['dataset'],r['method'],r['seed'])].append(r)
    summary=[dict(dataset=d,method=m,seed=seed,scene_groups=len(v),**{k:statistics.mean(x[k] for x in v) for k in keys}) for (d,m,seed),v in sorted(macros.items())]
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'per_recording.csv',video); write_csv(out/'per_scene.csv',scene); write_csv(out/'scene_macro.csv',summary); write_csv(out/'P_vs_Ofast_per_recording.csv',comparison)
    write_json(out/'summary.json',dict(status='PARTIAL_ANALYSIS',protocol_sha256=sha(protocol),frames_sha256=sha(frames_path),timing_sha256=sha(timing_path),warning='Paired valid-frame descriptive estimates; inspect failure counts. No bootstrap, call counts, timing distributions or temporal metric computed.'))
    print('Analysis written:',out)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('freeze'); a.add_argument('--config',required=True); a.add_argument('--out',required=True)
    a=sub.add_parser('report'); a.add_argument('--protocol',required=True); a.add_argument('--frames',required=True); a.add_argument('--timing',required=True); a.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='freeze': freeze(a.config,a.out)
    else: aggregate(a.protocol,a.frames,a.timing,a.out)

if __name__=='__main__': main()
