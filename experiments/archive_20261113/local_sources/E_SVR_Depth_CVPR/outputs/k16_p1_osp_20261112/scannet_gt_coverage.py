import csv, io, json
from pathlib import Path
import h5py, numpy as np
from PIL import Image

out=Path(r'E:\SVR-Depth-CVPR\outputs\k16_p1_osp_20261112\scannet_gt_candidates.csv')
rows=[]
with h5py.File(r'D:\SVR-Depth-data\VeloDepth-Evaluation\ScanNet.hdf5','r') as h:
    seq=json.loads(bytes(h['sequences_.json'][:]).decode())
    for name,s in sorted(seq.items()):
        if not name.endswith('_val'): continue
        flags=[]; errors=[]
        for i,key in enumerate(s['depth']):
            try:
                a=np.array(Image.open(io.BytesIO(h[key][:].tobytes())))
                flags.append(bool(np.any((a>10)&(a<10000))))
            except Exception as e:
                flags.append(False); errors.append(f'{i}:{e!r}')
        cycles=[i//16 for i in range(0,len(flags)-15,16) if all(flags[i:i+16])]
        rows.append({'sequence_id':name,'depth_count':len(flags),'valid_gt_frames':sum(flags),'gt_decode_or_empty_frames':len(flags)-sum(flags),'gt_complete_index_cycle_count':len(cycles),'gt_complete_index_cycles':';'.join(map(str,cycles)),'eligible_cycle_count':0,'time_continuity':'UNKNOWN','note':'index-aligned GT candidates only; not temporal eligibility','errors':'; '.join(errors)})
with out.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('sequences',len(rows),'frames',sum(r['depth_count'] for r in rows),'valid_frames',sum(r['valid_gt_frames'] for r in rows),'index_candidates',sum(r['gt_complete_index_cycle_count'] for r in rows),'errors',sum(bool(r['errors']) for r in rows),flush=True)
