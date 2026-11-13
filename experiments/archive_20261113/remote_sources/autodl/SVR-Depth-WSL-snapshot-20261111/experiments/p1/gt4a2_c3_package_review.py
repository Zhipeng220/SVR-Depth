#!/usr/bin/env python3
"""Compact whitelist packager; never include cycle_cache or pretrained backbone."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/gt4a2_c3'
SMOKE=ROOT/'outputs/gt4a2_c3_smoke'
ARCH=ROOT/'outputs/GT4_A2_C3_review.zip'
OUT.mkdir(parents=True,exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

required=[
    'gt4a2_c3_preflight.json','gt4a2_c3_config.json',
    'gt4a2_c3_C_training_curve.csv','gt4a2_c3_D_training_curve.csv',
    'gt4a2_c3_C_training_summary.json','gt4a2_c3_D_training_summary.json',
    'gt4a2_c3_full_sequence_frame_metrics.csv','gt4a2_c3_sequence_summary.csv',
    'gt4a2_c3_cycle_summary.csv','gt4a2_c3_crowd2_summary.json',
    'gt4a2_c3_keyframe_checks.csv','gt4a2_c3_decision.json',
    'gt4a2_c3_summary.json','GT4_A2_C3_report.md',
]
optional=['c3_preflight.log','c3_train_eval.log',
          'gt4a2_c3_sample_order.csv','gt4a2_c3_C_final_step1000.pt',
          'gt4a2_c3_D_final_step1000.pt','gt4a2_c3_initial_head.pt']
source=[
    'experiments/p1/gt4a2_c3_closed_loop.py',
    'experiments/p1/run_gt4a2_c3.sh',
    'experiments/p1/gt4a2_c3_package_review.py',
    'experiments/p1/gt4a2_exp1_train_eval.py',
    'experiments/p1/gt2_fixed_clip_repair_train.py',
    'velodepth/models/decoder_velo.py',
    'velodepth/models/persistent_feature_repair.py',
    'GT4_A2_C3_README.md',
]
entries={}
missing=[]
for name in required:
    p=OUT/name
    if p.is_file(): entries[str(p.relative_to(ROOT))]=p
    else: missing.append(str(p.relative_to(ROOT)))
for name in optional:
    p=OUT/name
    if p.is_file():entries[str(p.relative_to(ROOT))]=p
for name in source:
    p=ROOT/name
    if p.is_file():entries[name]=p
    else:missing.append(name)
for name in ('c3_smoke.log','gt4a2_c3_smoke_summary.json'):
    p=SMOKE/name
    if p.is_file():entries[str(p.relative_to(ROOT))]=p
manifest={
    'experiment':'GT4-A2-C3 closed-loop repair training',
    'package_status':'COMPLETE' if not missing else 'INCOMPLETE',
    'missing_required_files':missing,
    'excluded':['cycle_cache','feature states','pretrained model.safetensors',
                'recovery optimizer checkpoints','raw RGB/GT datasets','full depth predictions'],
    'files':[
      {'archive_path':name,'size':p.stat().st_size,'sha256':sha(p)}
      for name,p in sorted(entries.items())]}
manifest_path=OUT/'gt4a2_c3_package_manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
entries[str(manifest_path.relative_to(ROOT))]=manifest_path
ARCH.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(ARCH,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in sorted(entries.items()): z.write(p,arcname=name)
with zipfile.ZipFile(ARCH) as z:
    bad=z.testzip()
    if bad:raise RuntimeError(f'Zip integrity failure: {bad}')
print('PACKAGE STATUS:',manifest['package_status'])
print('MISSING:',missing)
print('ARCHIVE:',ARCH)
print('SIZE MiB:',round(ARCH.stat().st_size/1024**2,3))
print('ZIP integrity: PASS')
raise SystemExit(0 if not missing else 1)
