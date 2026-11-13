import csv, json
from pathlib import Path
import h5py

p=Path(__file__).resolve().parent
h5path=Path(r'D:\SVR-Depth-data\VeloDepth-Evaluation\TUM.hdf5')
with (p.parent/'P0_TUM_working_inventory.csv').open(encoding='utf-8-sig',newline='') as f:
    inventory={r['sequence_id']:r for r in csv.DictReader(f)}
frames=[]; seqrows=[]
with h5py.File(h5path,'r') as h:
    seq=json.loads(bytes(h['sequences.json'][:]).decode('ascii'))
    keys=[x.rsplit(' ',1)[0] for x in bytes(h['val.txt'][:]).decode('ascii').splitlines()]
    for key in keys:
        s=seq[key]; images=s['image']; depths=s['depth']; old=inventory[key]
        same=0
        for i,(im,dep) in enumerate(zip(images,depths)):
            match=Path(im).name.removesuffix('_c.jpg')==Path(dep).name.removesuffix('_d.png')
            same+=match
            frames.append(dict(sequence_id=key,hdf5_index=i,cycle_index=i//16,age=i%16,
                rgb_hdf5_key=im,gt_hdf5_key=dep,storage_label_match=match,
                original_rgb_frame_id='UNKNOWN',original_gt_frame_id='UNKNOWN',
                rgb_timestamp='UNKNOWN',gt_timestamp='UNKNOWN',rgb_gt_delta_seconds='UNKNOWN',
                rgb_interframe_delta_seconds='UNKNOWN',gt_interframe_delta_seconds='UNKNOWN',
                mapping_status='UNKNOWN: conversion/source mapping absent'))
        seqrows.append(dict(sequence_id=key,domain='TUM',frame_count=len(images),
            rgb_gt_storage_label_matches=f'{same}/{len(images)}',original_rgb_frame_ids_recovered=0,
            original_gt_frame_ids_recovered=0,rgb_timestamps_recovered=0,gt_timestamps_recovered=0,
            rgb_gt_pairing_certified='NO',temporal_continuity_certified='NO',
            physical_scene_independence_certified='NO',physical_scene_id=old['physical_scene_id'],
            gt_complete_index_cycle_candidates=old['gt_complete_index_cycle_candidates'],
            certified_cycles=0,status='INDEX_CANDIDATE_ONLY; INSUFFICIENT_EVIDENCE',
            exclusion_reason='Original RGB/GT timestamps, conversion mapping and independent physical scene identity not established',
            exposure_status=old['exposure_status'],source='TUM.hdf5 val.txt and sequences.json; prior P0 inventory'))

for name,rows in [('P01_TUM_TEMPORAL_MAPPING.csv',seqrows),('P01_TUM_FRAME_MAPPING.csv',frames)]:
    with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
print('sequences',len(seqrows),'frames',len(frames),'matched storage labels',sum(r['storage_label_match'] for r in frames))
