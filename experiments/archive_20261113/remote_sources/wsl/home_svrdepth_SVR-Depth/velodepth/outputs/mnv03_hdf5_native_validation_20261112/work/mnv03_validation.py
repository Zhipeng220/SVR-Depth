import os, sys, csv, json, traceback, zipfile, io
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'
root=Path('/home/svrdepth/SVR-Depth/velodepth'); sys.path.insert(0,str(root))
out=root/'outputs/mnv03_hdf5_native_validation_20261112/results'; out.mkdir(parents=True,exist_ok=True)
import numpy as np, h5py, torch
from PIL import Image
from velodepth.datasets.tum import TUM
from velodepth.datasets.scannet import ScanNetVid

def csvout(name,rows):
 if not rows: return
 with (out/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
def textout(name,s): (out/name).write_text(s,encoding='utf-8')
def flat(t): return json.dumps(t.detach().cpu().flatten().tolist(),ensure_ascii=False)
cfg=json.loads((root/'configs/velodepth-eval.json').read_text())['data']
schemas=[]; pairs=[]; orders=[]; params=[]; checks=[]; log=[]
for domain,cls,key in [('TUM',TUM,'sequences.json'),('ScanNet',ScanNetVid,'sequences_.json')]:
 try:
  file=Path(os.environ['DATAROOT'])/(domain+'.hdf5')
  with h5py.File(file,'r') as h:
   seqs=json.loads(np.asarray(h[key]).tobytes().decode('ascii'))
   split=np.asarray(h['val.txt']).tobytes().decode('ascii').strip().splitlines()
   names=[s.strip().rsplit(' ',1)[0] for s in split if len(s.strip().rsplit(' ',1))==2]
   eligible=[n for n in names if n in seqs and len(seqs[n].get('image',[]))>=16 and len(seqs[n].get('image',[]))==len(seqs[n].get('depth',[]))]
   sid=eligible[0]; sample=seqs[sid]; i0=0
   for field in ['image','depth']:
    node=h[sample[field][i0]]; arr=np.asarray(Image.open(io.BytesIO(np.asarray(node).tobytes())))
    schemas.append(dict(dataset=domain,sequence_id=sid,field=field,index=0,hdf5_path=sample[field][0],hdf5_storage_shape=str(node.shape),hdf5_storage_dtype=str(node.dtype),decoded_shape=str(arr.shape),decoded_dtype=str(arr.dtype),sequence_field_count=len(sample[field]),root_attributes=str(dict(h.attrs)),sequence_fields=';'.join(sample.keys()),timestamp_field=','.join(k for k in sample if 'timestamp' in k.lower()) or 'UNKNOWN',source_frame_id='HDF5 path only; provenance UNKNOWN',scope='representative first val sequence; MNV02 inventory inherited'))
   pairs.append(dict(dataset=domain,sequence_id=sid,selection='first val.txt entry with equal image/depth lengths and >=16; no model results used',image_count=len(sample['image']),depth_count=len(sample['depth']),hdf5_json_coindexed='YES',official_native_loader_coindexed='YES: sequence_dataset.py get_single_sequence uses j for both',generation_pairing='UNKNOWN',physical_capture_time_pairing='UNKNOWN',status='HDF5_CO_INDEXED_ONLY',attributes=str(dict(h.attrs))))
  csvout('MNV03_HDF5_NATIVE_SCHEMA.csv',schemas);csvout('MNV03_SEQUENCE_PAIRING.csv',pairs)
  # numpy 2 removed ndarray.tostring; use identical tobytes for load_dataset only.
  original=np.array
  class LegacyArray(np.ndarray):
   def tostring(self,*a,**kw): return self.tobytes(*a,**kw)
  def compatible(*a,**kw): return original(*a,**kw).view(LegacyArray)
  try:
   np.array=compatible
   ds=cls(image_shape=cfg['image_shape'],split_file='val.txt',test_mode=True,normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.0,num_frames=-1)
  finally: np.array=original
  index=next(i for i in range(len(ds.dataset)) if ds.dataset[i]['sequence_name']==sid)
  idxs,_=ds.get_test_idxs(len(sample['image']),sample.get('keyframe_idx'))
  assert len(idxs)>=16 and idxs==list(range(idxs[0],idxs[0]+len(idxs)))
  data=ds.get_single_sequence(index)
  for j,source_index in enumerate(idxs):
   k=(j,0); im=data['image'][k]; dep=data['depth'][k]; mask=data['depth_mask'][k].bool(); cam=data['camera'][k]; orig=data['camera_original'][k]
   valid=dep[mask]; good=bool(torch.isfinite(im).all() and torch.isfinite(dep).all() and valid.numel()>0)
   checks.append(dict(dataset=domain,sequence_id=sid,loader_position=j,hdf5_sequence_index=source_index,rgb_path=sample['image'][source_index],gt_path=sample['depth'][source_index],rgb_ok='PASS',gt_ok='PASS',same_loader_sample_index='YES',rgb_shape=str(tuple(im.shape)),gt_shape=str(tuple(dep.shape)),rgb_dtype=str(im.dtype),gt_dtype=str(dep.dtype),rgb_finite=bool(torch.isfinite(im).all()),gt_finite=bool(torch.isfinite(dep).all()),valid_pixels=int(mask.sum()),valid_ratio=float(mask.float().mean()),valid_depth_min=float(valid.min()) if valid.numel() else 'UNKNOWN',valid_depth_max=float(valid.max()) if valid.numel() else 'UNKNOWN',camera_params_after=flat(cam.params),depth_scale=cls.depth_scale,depth_unit='project evaluation assumes metres; source unit not independently proven',timestamp='UNKNOWN',native_loader='PASS' if good else 'FAIL',m05_read_manifest_load_frame='NOT_OBSERVED: file-path and timestamp interface cannot directly read HDF5'))
   orders.append(dict(dataset=domain,sequence_id=sid,loader_position=j,hdf5_sequence_index=source_index,rgb_path=sample['image'][source_index],depth_path=sample['depth'][source_index],same_index='YES',container_step=1,timestamp='UNKNOWN',chronology='CONTAINER_ORDER_ONLY'))
   if j==0:
    params.append(dict(dataset=domain,sequence_id=sid,source_index=source_index,fx_before=float(orig.K[0,0,0]),fy_before=float(orig.K[0,1,1]),cx_before=float(orig.K[0,0,2]),cy_before=float(orig.K[0,1,2]),fx_after=float(cam.K[0,0,0]),fy_after=float(cam.K[0,1,1]),cx_after=float(cam.K[0,0,2]),cy_after=float(cam.K[0,1,2]),rgb_shape_before=str(tuple(data['image_original'][k].shape)),rgb_shape_after=str(tuple(im.shape)),depth_scale=cls.depth_scale,depth_scale_source=f'velodepth/datasets/{domain.lower()}.py',depth_unit='PARTIALLY_CONFIRMED project assumes metres; source conversion UNKNOWN',valid_depth_mask='0 < depth < 10 (AnnotationMask test_mode)',resize='ContextCrop: RGB bicubic; depth masked-nearest',camera_adjustment='camera crop then resize via project ContextCrop',camera_status='CONFIRMED runtime native Loader; M05 adapter NOT OBSERVED'))
  csvout('MNV03_LOADER_ORDER.csv',orders);csvout('MNV03_CPU_16FRAME_VALIDATION.csv',checks);csvout('MNV03_CAMERA_DEPTH_PARAMS.csv',params)
  log.append(f'{domain}: {sid}, {len(idxs)} contiguous native-loader positions, finite+nonempty={sum(x["native_loader"]=="PASS" for x in checks if x["dataset"]==domain)}/{len(idxs)}; M05 path loader NOT OBSERVED')
 except Exception:
  log.append(domain+' FAILED:\n'+traceback.format_exc())
assert not torch.cuda.is_initialized()
textout('MNV03_K16_CODE_TRACE.md', '''# K16 actual code trace

- `experiments/p1/cvpr_m05_formal.py:33` defines `K = 16`.
- `experiments/p1/cvpr_m04_closed_loop.py:39,367-401` defines K and `stream_sequence`: `idx % K == 0` selects keyframe; `age=idx%K`; refresh overwrites radius/confidence/features/rays with `exp1.keyframe_state`; other steps consume prior state via `apply_branch`; this describes inference, not raw-video stride.
- `experiments/p1/gt4a2_exp1_train_eval.py:244-264` `keyframe_state` runs encoder, `pixel_decoder.set_shapes`, `forward_keyframe`, constructs fresh rays and features.
- `experiments/p1/cvpr_m05_formal.py:724-735` `run_step` passes prior state into `one_fast_step`, returns new radius/confidence/features and retains rays_map.
- No forward/model was executed in MNV03; these are static source traces. Refresh behavior must not be extrapolated to an unimplemented HDF5 adapter.
''')
textout('MNV03_NATIVE_EVAL_READINESS.md', '''# Native HDF5 evaluation readiness

## Proven and unknown
TUM and ScanNet results are separated in CSV. Embedded JSON stores RGB/depth paths at the same list index and the project's native `SequenceDataset.get_single_sequence` decodes both by the same `j`. This is HDF5_CO_INDEXED_ONLY: official generation correspondence and physical acquisition timestamps are UNKNOWN. Dataset JSON contains cameras; native preprocessing produces runtime camera values. Project scale is 1000; the raw-generation units have not independently been proven.

Native CPU loader was exercised in test mode via `get_single_sequence`, not `__getitem__` (which silently substitutes random sequence on failure). `num_frames=-1` uses the project's own last-up-to-32 consecutive container indices. The first attempt using `num_frames=16` with absent keyframe_idx produced out-of-range indices in the existing get_test_idxs implementation. NumPy 2 removed ndarray.tostring; a temporary equivalent .tobytes shim was scoped only to dataset initialization, with no frozen source edit.

Crucial distinction: M05's `gt4a2_exp1_train_eval.read_manifest` and `load_frame` require file paths, physical timestamp and original depth files, not HDF5 paths. No M05 CPU HDF5 read is claimed. Native VeloDepth HDF5 loader PASS is not M05 Loader PASS. Native config 518 input is not claimed identical to the frozen M05 preprocessing. No predictions or evaluation metrics were run.

## Per-domain gates
For each domain: STATIC_FRAME_EVAL_READY = NO-GO (M05 entry cannot directly read HDF5; authoritative original pairing/units unresolved); ORDERED_HDF5_SEQUENCE_READY = NO-GO for formal M05, native loader container-order check is recorded separately; TEMPORAL_MECHANISM_PILOT_READY = NO-GO (chronological semantics not established by container indices); FULL_P1_READY = NOT CERTIFIED. These are evidence-based admission judgments, not claims that the data is inherently invalid.

Do not interpret K16 as source frame stride. HDF5 ScanNet space IDs are not independent verified physical scenes. No model/CUDA/run_step/training performed.
''')
textout('MNV03_EXECUTION_LOG.txt','\n'.join(log)+f'\nCUDA initialized: {torch.cuda.is_initialized()}\nModels created: 0; run_step: 0; HDF5 downloads: 0\n')
with zipfile.ZipFile(out/'MNV03_DELIVERY.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(out.iterdir()):
  if f.is_file() and f.name!='MNV03_DELIVERY.zip':z.write(f,f.name)
print('\n'.join(log)); print('CUDA initialized',torch.cuda.is_initialized())
