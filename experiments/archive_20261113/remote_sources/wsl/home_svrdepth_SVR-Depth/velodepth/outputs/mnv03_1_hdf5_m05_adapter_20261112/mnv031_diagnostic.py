import os,sys,csv,json,io,traceback,zipfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'
root=Path('/home/svrdepth/SVR-Depth/velodepth');sys.path[:0]=[str(root),str(root/'experiments/p1')]
out=root/'outputs/mnv03_1_hdf5_m05_adapter_20261112';out.mkdir(parents=True,exist_ok=True)
import numpy as np, torch, h5py, tables
from PIL import Image
import gt2_fixed_clip_repair_train as gt2
from velodepth.datasets.utils_decode import decode_rgb,decode_depth
from velodepth.models.velodepth import _postprocess

def writecsv(name,rows):
 if rows:
  with (out/name).open('w',encoding='utf8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def write(name,s): (out/name).write_text(s,encoding='utf8')
prior=root/'outputs/mnv03_hdf5_native_validation_20261112/results/MNV03_CPU_16FRAME_VALIDATION.csv'
with prior.open(newline='',encoding='utf8') as f:old=list(csv.DictReader(f))
cfg=json.loads((Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c/config.json').read_text())
sc=cfg['data']['shape_constraints'];ratio=sc['ratio_bounds'];lo=float(sc['pixels_min']);hi=float(sc['pixels_max']);bounds=(lo,lo+(hi-lo)/10)
mean=torch.tensor(gt2.IMAGENET_DATASET_MEAN).view(1,3,1,1);std=torch.tensor(gt2.IMAGENET_DATASET_STD).view(1,3,1,1)
checks=[];align=[];pre=[];logs=[]
for domain,key in [('TUM','sequences.json'),('ScanNet','sequences_.json')]:
 try:
  existing=[r for r in old if r['dataset']==domain];assert len(existing)==32
  indices=[int(r['hdf5_sequence_index']) for r in existing];assert indices==list(range(indices[0],indices[0]+32))
  path=Path(os.environ['DATAROOT'])/(domain+'.hdf5')
  with h5py.File(path,'r') as h:
   seqs=json.loads(np.asarray(h[key]).tobytes().decode('ascii')); sid=existing[0]['sequence_id'];sample=seqs[sid]
   assert all(sample['image'][int(r['hdf5_sequence_index'])]==r['rgb_path'] and sample['depth'][int(r['hdf5_sequence_index'])]==r['gt_path'] for r in existing)
  with tables.open_file(path,'r') as h:
   for pos in [0,16,31]:
    r=existing[pos]; idx=int(r['hdf5_sequence_index']);result={'image_fields':set(),'gt_fields':set(),(0,0):{'image_fields':set(),'gt_fields':set()}}
    decode_rgb(result,h,sample['image'][idx],0);decode_depth(result,h,sample['depth'][idx],0,depth_scale=1000.0)
    rgb=result[(0,0)]['image'];depth=result[(0,0)]['depth'];arr=rgb[0].permute(1,2,0).numpy().copy();hw=tuple(arr.shape[:2])
    pads,padded=gt2.get_paddings(hw,ratio);_,net=gt2.get_resize_factor(padded,bounds)
    x,u8=gt2.preprocess_rgb(arr,torch.device('cpu'),pads,net,mean,std)
    mock=torch.zeros((1,3,*net)); restored=_postprocess(mock,padded,pads,interpolation_mode='bilinear')
    camera=sample.get('camera_params',sample.get('K'))[idx]
    checks.append(dict(dataset=domain,sequence_id=sid,logical_step=pos,hdf5_index=idx,physical_timestamp='UNKNOWN',rgb_path=r['rgb_path'],gt_path=r['gt_path'],native_decoder='PASS',m05_preprocess='PASS',m05_formal_manifest='BLOCKED: requires physical timestamp and external file paths',m05_rgb_shape=str(tuple(x.shape)),m05_rgb_dtype=str(x.dtype),frame_u8_shape=str(tuple(u8.shape)),original_rgb_shape=str(tuple(rgb.shape)),decoded_gt_shape=str(tuple(depth.shape)),gt_dtype=str(depth.dtype),same_source_index='YES',depth_scale=1000,source_depth_unit='UNKNOWN',gt_mask_positive_pixels=int((depth>0).sum()),camera_original=json.dumps(camera),finite=bool(torch.isfinite(x).all() and torch.isfinite(depth).all()),status='PARTIAL'))
    align.append(dict(dataset=domain,sequence_id=sid,logical_step=pos,hdf5_index=idx,source_rgb_hw=str(hw),source_depth_hw=str(tuple(depth.shape[-2:])),same_dimensions=hw==tuple(depth.shape[-2:]),m05_padding=str(pads),m05_network_hw=str(net),m05_padded_hw=str(padded),postprocess_result_hw=str(tuple(restored.shape[-2:])),postprocess_geometry='resize to padded canvas then remove padding; synthetic shape-only check',source_registration='UNKNOWN',pixel_correspondence='UNPROVEN',status='NO-GO'))
  for r in existing:
   pre.append(dict(dataset=domain,sequence_id=sid,logical_step=r['loader_position'],hdf5_index=r['hdf5_sequence_index'],rgb_path=r['rgb_path'],gt_path=r['gt_path'],prior_native_rgb_shape=r['rgb_shape'],prior_native_gt_shape=r['gt_shape'],prior_native_status=r['native_loader'],native_valid_ratio=r['valid_ratio'],native_test_transform='ContextCrop keep_original: RGB crop+resize; GT left original; camera crop+resize',m05_transform='normalize RGB, pad, bilinear resize; GT original; postprocess restores padded canvas then unpads',physical_timestamp='UNKNOWN',order='CONTAINER_ORDER_ONLY',m05_adapter='NOT_IMPLEMENTED: protocol unproven'))
  writecsv('MNV031_PREPROCESS_COMPARISON.csv',pre);writecsv('MNV031_RGB_GT_ALIGNMENT.csv',align);writecsv('MNV031_ADAPTER_CPU_RESULTS.csv',checks)
  logs.append(f'{domain}: reused MNV03 32 indices; direct official decoders and M05 preprocess checked for positions 0,16,31. M05 formal Loader NOT RUN (HDF5 path+timestamps unsupported).')
 except Exception:logs.append(domain+' ERROR: '+traceback.format_exc())
assert not torch.cuda.is_initialized()
write('MNV031_M05_INPUT_PROTOCOL.md','''# M05 输入协议

- `experiments/p1/gt4a2_exp1_train_eval.py:84-139` read_manifest 读取 CSV，按 frame_index_zero_based 排序且检查从零连续、真实 RGB 文件路径、递增 rgb_timestamp；MATCHED 还要求真实 depth 文件、唯一 GT 和有界 abs_delta_ms。
- `cvpr_m05_formal.py:155-216` 正式 manifest 同样要求这些字段、有限递增时间戳和 RGB/GT 时间差；它不是仅记录序号的接口。数据未提供物理时间与配对误差，不得填 logical_step 冒充。
- `gt4a2_exp1_train_eval.py:177-240` shape_info_for 需要 model.shape_constraints 和首帧真实 RGB 路径；load_frame 要求 row、shape、device、need_gt，返回 index、rgb_path、rgb_file、rgb_timestamp、img (1,3,Hn,Wn) float32、frame_u8 (3,Hn,Wn) uint8、gt_np (H,W) float32、gt (H,W) float32 及 GT 路径/匹配信息。没有在其 frame 中传入 camera 参数；shape 有 validity_mask、paddings 等。
- M05 `run_step` 位于 `cvpr_m05_formal.py:724-735`，调用 `one_fast_step`（`gt4a2_exp1_train_eval.py:290-295`）。`transition_static:267-286` 使用前后帧 `frame_u8` / `img` 构建光流、归一化网格和图像残差；`gt2_fixed_clip_repair_train.py:365-398` 使用当前 `img`、static 和显式 feature/radius。该计算链未直接读取 timestamp；这不免除上游正式 manifest 的物理时间与 GT 配对校验。
- 正式原始 GT 以 uint16 / 5000 解释为米，不能直接用于 HDF5 的 /1000 文件。M05 评价 `gt2_fixed_clip_repair_train.py:121-138` 使用 GT>0、有限 GT/预测和预测>0；原生 Loader 的 test mask 是 0<depth<10，两者不是自动等价。
''')
write('MNV031_DEPTH_UNIT_VALIDATION.md','''# 深度尺度和单位

TUM 与 ScanNet 原生类各规定 `depth_scale=1000.0`，`utils_decode.py:21-46` 解码后除以 1000。M05 原始 GT 路径在 `gt2_fixed_clip_repair_train.py:42` 使用 /5000，其数据表示不同，不能混用。HDF5 生成转换及原始物理单位本轮未被独立证实，DEPTH_METRIC_UNIT_READY=NO-GO；不要把数值除以 1000 的实现等价为对数据生成单位的验证。原生与 M05 有效 mask 口径不同，真实评测必须事先对齐且不能通过改 GT 凑形状。
''')
write('MNV031_TEMPORAL_SEMANTICS.md','''# 时间与顺序

继承 MNV03 逐位置真实 Loader 证据并对本轮同一序列 JSON 重新核对 32 个连续索引及 image/depth 路径。结论 CONTAINER_ORDER_ONLY；物理时间戳 UNKNOWN，不能从容器索引推导真实采样周期或因果时间顺序。M05 低层单步计算没直接读取时间戳，但正式 read_manifest 显式检查时间和配对差，因此不能用 logical_step 替代正式字段。Chronological order 尚未得到生成代码、数据说明或真实元数据的证实；TEMPORAL_SEMANTICS_READY=NO-GO。
''')
write('MNV031_FINAL_DECISION.md','''# MNV-03.1 结论

两个域各保留并核对 MNV03 的 32 个原生 Loader 位置，首、中、末三个位置实际调用官方 RGB/Depth 解码和冻结 M05 的 RGB 预处理（CPU，非 M05 model）；结果见 CSV。

| 准入项 | TUM | ScanNet |
|---|---|---|
| HDF5_NATIVE_LOADER_READY | PASS（继承 MNV03） | PASS（继承 MNV03） |
| M05_INPUT_ADAPTER_READY | NO-GO | NO-GO |
| RGB_GT_PIXEL_ALIGNMENT_READY | NO-GO | NO-GO |
| DEPTH_METRIC_UNIT_READY | NO-GO | NO-GO |
| SEQUENCE_ORDER_READY | PASS，仅容器顺序 | PASS，仅容器顺序 |
| TEMPORAL_SEMANTICS_READY | NO-GO | NO-GO |
| MNV04_PILOT_READY | NO-GO | NO-GO |

原生 test-mode ContextCrop 的 keep_original 路径会对 RGB crop/resize，但保留 GT 原尺寸；MNV03 的 476×630 对 480×640 因而不能直接逐像素比较。M05 推理先 pad/resize 输入，再用 `_postprocess` 将预测插值回 padded canvas 并去 padding（`velodepth/models/velodepth.py:144-155`），以原始图像网格比较原始 GT。这个几何链和本轮 synthetic shape check 只证明输出形状的回映射，不证明 HDF5 内 RGB/GT 实际跨传感器配准或 HDF5 数据生成的像素对应。Camera 的原始和变换参数参见 MNV03 既有 CSV；本轮三个采样点报告原始 camera 数值。

完整 M05 输入协议、物理时间、原始 GT 配准与深度单位未获证明；按任务条件未实现 `HDF5M05Adapter`，以免制造具有误导性的兼容性。TUM 和 ScanNet 均暂不允许进入冻结 M05 正式评测；静态 RGB/GT 指标及 O/S/P 时序 Pilot 暂不准入。需取得权威数据生成/配准和单位说明、可验证的时序来源，并设计经批准的 M05 HDF5 接口或明确版本化协议；不得伪造 timestamp、调整 GT 尺寸来掩盖未知。
''')
write('MNV031_EXECUTION_LOG.txt','\n'.join(logs)+f'\nCUDA initialized: {torch.cuda.is_initialized()}\nModels created: 0; run_step calls: 0; GPU inference: 0; new downloads: 0\n')
with zipfile.ZipFile(out/'MNV031_DELIVERY.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(out.iterdir()):
  if f.is_file() and f.name!='MNV031_DELIVERY.zip':z.write(f,f.name)
print('\n'.join(logs));print('CUDA initialized:',torch.cuda.is_initialized())
