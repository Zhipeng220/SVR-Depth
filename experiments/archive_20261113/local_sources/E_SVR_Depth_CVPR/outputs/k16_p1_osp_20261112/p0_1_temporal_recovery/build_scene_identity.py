import csv
from pathlib import Path

out = Path(__file__).resolve().parent
with (out.parent / 'P0_SCENE_INVENTORY.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
fields = ['domain','sequence_id','physical_scene_id','candidate_group','same_room_or_repeat_scan',
          'A_temporal_continuity','B_rgb_gt_correspondence','C_independent_physical_scene',
          'D_index_candidates','E_evidence_insufficient','exposure_status','evidence']
result=[]
for r in rows:
    seq=r['sequence_id']
    if r['domain']=='TUM':
        group=r['physical_scene_id'] if r['physical_scene_id']!='UNKNOWN' else 'UNKNOWN'
        relation=('PROVISIONAL: desk/desk2/room same office area in official descriptions; local source mapping unverified'
                  if group!='UNKNOWN' else 'UNKNOWN')
        evidence='P0 TUM HDF5 metadata; docs/data_audit/PROVENANCE_FOLLOWUP/PROVENANCE_FOLLOWUP_REPORT.md'
    else:
        group=r['scene_scan_id']
        relation='train/val share scan label; HDF5-to-original scan mapping unverified'
        evidence='P0 ScanNet HDF5 metadata; official ScanNet data organization; PROVENANCE_FOLLOWUP_REPORT.md'
    exposure=r['exposure_status']
    if r['domain']=='TUM' and 'walking_halfsphere' in seq:
        exposure += '; historical ENG-09 exposure'
    result.append(dict(domain=r['domain'],sequence_id=seq,physical_scene_id='UNKNOWN',candidate_group=group,
        same_room_or_repeat_scan=relation,A_temporal_continuity='UNKNOWN',B_rgb_gt_correspondence='UNKNOWN',
        C_independent_physical_scene='UNKNOWN',D_index_candidates=r['gt_complete_index_cycle_candidates'],
        E_evidence_insufficient='YES',exposure_status=exposure,evidence=evidence))
with (out/'P01_SCENE_IDENTITY.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(result)
print('rows',len(result))
