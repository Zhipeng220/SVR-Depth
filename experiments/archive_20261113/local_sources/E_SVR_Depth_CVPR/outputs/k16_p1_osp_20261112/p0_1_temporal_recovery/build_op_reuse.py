"""Summarize existing V5-C frame records; no model or hashes computed."""
import csv, io, json, zipfile
from collections import defaultdict
from pathlib import Path

root = Path(r'E:\SVR-Depth-CVPR')
archive = root / 'SVR-Depth_实验证据快照_20261109.zip'
out = Path(__file__).parent / 'P01_HISTORICAL_OP_REUSE.csv'
base = '01_V4_V5/outputs/y04_v5c_external_20261109_171333/'
rows = []
with zipfile.ZipFile(archive) as z:
    for domain in ('TUM', 'ScanNet'):
        protocol = json.loads(z.read(base + domain + '/protocol.json'))
        groups = defaultdict(lambda: {'O': {}, 'P': {}})
        for arm in ('O','P'):
            with z.open(base + domain + '/v5_' + arm + '.csv') as f:
                for rec in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')):
                    groups[rec['sequence']][arm][int(rec['index'])] = rec
        for seq in protocol['sequence_ids']:
            g=groups[seq]; both=sorted(set(g['O']) & set(g['P']))
            complete = 0
            for start in range(0, max(both, default=-1) + 1, 16):
                if all(i in g['O'] and i in g['P'] and g['O'][i]['status']=='ok' and g['P'][i]['status']=='ok' for i in range(start,start+16)):
                    complete += 1
            rows.append(dict(domain=domain,sequence_id=seq,historical_O_frames=len(g['O']),historical_P_frames=len(g['P']),paired_index_frames=len(both),
                complete_index_cycles=complete,age_mapping='index mod 16; keyframe index mod 16 == 0 in historical script',
                raw_frame_id='NOT_SAVED in O/P CSV; index only; HDF5 mapping requires provenance',
                predicted_depth='NOT_FOUND in inspected V5-C archive result directory',gt_pixels='NOT_SAVED in O/P CSV',
                pixel_mask='NOT_SAVED; per-frame valid count and mask fingerprint only',
                shared_support_recalculable='NO; pixel predictions/masks absent',
                protocol_compatibility='UNVERIFIED: data provenance and shared S support unresolved',
                reusable_p1_cycles=0,reason='No pixel-level O/P outputs for re-evaluation; current local input identity and common S support unverified',
                source=str(archive)+' :: '+base+domain+'/v5_{O,P}.csv; protocol.json'))
fields=list(rows[0])
with out.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
for domain in ('TUM','ScanNet'):
    r=[x for x in rows if x['domain']==domain]
    print(domain,'sequences',len(r),'O frames',sum(x['historical_O_frames'] for x in r),'P frames',sum(x['historical_P_frames'] for x in r),'complete index cycles',sum(x['complete_index_cycles'] for x in r))
