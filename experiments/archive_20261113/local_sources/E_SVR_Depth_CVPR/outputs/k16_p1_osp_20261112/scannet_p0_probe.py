import csv, io, json, re
from pathlib import Path
import h5py, numpy as np
from PIL import Image

root=Path(r'E:\SVR-Depth-CVPR')
out=root/'outputs/k16_p1_osp_20261112/scannet_probe.csv'
history=Path(r'C:\Users\pyk18\Documents\ChatGPT\SVRDEPTH\V5_K16_配对周期结果.csv')
with history.open(encoding='utf-8-sig',newline='') as f:
    hist=list(csv.DictReader(f))
byseq={}
for r in hist:
    if r['dataset']=='ScanNet': byseq.setdefault(r['sequence'],[]).append(r)
with (root/'docs/data_audit/experiment_usage_history.csv').open(encoding='utf-8-sig',newline='') as f:
    usage={(r['dataset'],r['sequence_id']):r for r in csv.DictReader(f)}
rows=[]
with h5py.File(r'D:\SVR-Depth-data\VeloDepth-Evaluation\ScanNet.hdf5','r') as h:
    seq=json.loads(bytes(h['sequences_.json'][:]).decode())
    for name,s in sorted(seq.items()):
        ims=s.get('image',[]); depths=s.get('depth',[]); n=min(len(ims),len(depths)); sample=sorted(set([0,n//2,n-1])) if n else []
        paired=len(ims)==len(depths) and all(Path(a).stem==Path(b).stem for a,b in zip(ims,depths))
        present=all(k in h for k in ims+depths)
        ok=0; valid=0; err=''
        for i in sample:
            try:
                im=np.array(Image.open(io.BytesIO(h[ims[i]][:].tobytes())).convert('RGB'))
                gt=np.array(Image.open(io.BytesIO(h[depths[i]][:].tobytes())))
                ok+=1; valid+=bool(np.any((gt>10)&(gt<10000)))
            except Exception as e: err+=repr(e)+'; '
        u=usage.get(('ScanNet',name),{}); hh=byseq.get(name,[])
        rows.append(dict(sequence_id=name,scene_scan_id=name.rsplit('_',1)[0],physical_scene_id='UNKNOWN',domain='ScanNet',exposure_status=('DEVELOPMENT' if u.get('P2B')=='YES' or hh else 'UNKNOWN'),timestamp_status='UNKNOWN: numeric frame identifiers and sampled stride do not prove original timestamps or continuity',gt_coverage=f'{valid}/{len(sample)} sampled GT payloads contain valid depth; all-frame coverage UNKNOWN',complete_cycle_count='UNKNOWN',historical_OP_available=('YES_CYCLE_METRICS_ONLY' if hh else 'UNKNOWN'),eligibility='UNKNOWN',exclusion_reason='physical room independence and original timestamp/continuity unverified',images=len(ims),depths=len(depths),all_keys_present=present,paired_frame_stem_equal=paired,sample_pairs_decoded=ok,sample_count=len(sample),historical_paired_complete_cycles=sum(r['n_frames']=='16' and r['n_valid']=='16' for r in hh),error=err))
with out.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('rows',len(rows),'all keys',sum(r['all_keys_present'] for r in rows),'stem match',sum(r['paired_frame_stem_equal'] for r in rows),'sample decode complete',sum(r['sample_pairs_decoded']==r['sample_count'] for r in rows),'OP',sum(r['historical_OP_available'].startswith('YES') for r in rows),'errors',[(r['sequence_id'],r['error']) for r in rows if r['error']][:5])
