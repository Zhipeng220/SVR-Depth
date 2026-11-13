import csv
from pathlib import Path

p = Path(__file__).resolve().parent
def read(name):
    with (p / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

tum = read('P0_TUM_working_inventory.csv')
scan = read('scannet_probe.csv')
gt = {r['sequence_id']: r for r in read('scannet_gt_candidates.csv')}
fields = ['physical_scene_id', 'sequence_id', 'domain', 'exposure_status',
          'timestamp_status', 'gt_coverage', 'complete_cycle_count',
          'historical_OP_available', 'eligibility', 'exclusion_reason',
          'gt_complete_index_cycle_candidates', 'rgb_gt_readability', 'scene_scan_id']
rows = []
for r in tum:
    rows.append({k: r.get(k, '') for k in fields})
for r in scan:
    q = gt.get(r['sequence_id'])
    rows.append(dict(physical_scene_id=r['physical_scene_id'], sequence_id=r['sequence_id'],
        domain=r['domain'], exposure_status=r['exposure_status'], timestamp_status=r['timestamp_status'],
        gt_coverage=(f"{q['valid_gt_frames']}/{q['depth_count']} GT frames with valid pixels" if q else r['gt_coverage']),
        complete_cycle_count='0' if q else 'UNKNOWN (not fully checked)',
        historical_OP_available=r['historical_OP_available'] + '; exact P1 compatibility unverified',
        eligibility='NO', exclusion_reason=r['exclusion_reason'] + ('; train split not evaluated for P1' if not q else ''),
        gt_complete_index_cycle_candidates=q['gt_complete_index_cycle_count'] if q else 'UNKNOWN',
        rgb_gt_readability=f"{r['sample_pairs_decoded']}/{r['sample_count']} sampled pairs decoded; all referenced keys exist={r['all_keys_present']}",
        scene_scan_id=r['scene_scan_id']))
with (p/'P0_SCENE_INVENTORY.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
with (p/'P0_ELIGIBLE_CYCLES.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w=csv.writer(f); w.writerow(['domain','physical_scene_id','sequence_id','cycle_id','start_index','end_index','timestamp_status','gt_coverage','eligibility'])
print(f'Inventory: {len(tum)} TUM val and {len(scan)} ScanNet train/val; eligible cycles: 0')
