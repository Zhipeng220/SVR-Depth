import os,sys,csv,json,io,zipfile,traceback,re
from decimal import Decimal
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path('/home/svrdepth/SVR-Depth/velodepth');sys.path.insert(0,str(root))
out=root/'outputs/mnv03_2_official_protocol_recovery_20261112';out.mkdir(parents=True,exist_ok=True)
import numpy as np,h5py,torch
from PIL import Image
previous=root/'outputs/mnv03_hdf5_native_validation_20261112/results/MNV03_CPU_16FRAME_VALIDATION.csv'
with previous.open(newline='',encoding='utf8') as f: rows=list(csv.DictReader(f))
def csvout(name,data):
 if data:
  with (out/name).open('w',newline='',encoding='utf8') as f:
   w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
def report(name,text): (out/name).write_text(text,encoding='utf8')
pair=[];order=[];samples=[];log=[]
for domain in ['TUM','ScanNet']:
 try:
  r=[x for x in rows if x['dataset']==domain];assert len(r)==32
  p=Path('/home/svrdepth/datasets/velodepth_evaluation')/(domain+'.hdf5');key='sequences.json' if domain=='TUM' else 'sequences_.json'
  with h5py.File(p,'r') as h:
   data=json.loads(np.asarray(h[key]).tobytes().decode('ascii'));sid=r[0]['sequence_id'];s=data[sid]; nums=[]
   for i,x in enumerate(r):
    idx=int(x['hdf5_sequence_index']);rgb=s['image'][idx];gt=s['depth'][idx];assert rgb==x['rgb_path'] and gt==x['gt_path']
    a=Path(rgb).stem;b=Path(gt).stem
    a0=re.sub(r'_c$','',a) if domain=='TUM' else a;b0=re.sub(r'_d$','',b) if domain=='TUM' else b
    token=a0.replace('-','.') if domain=='TUM' else a0
    try:num=Decimal(token);nums.append(num)
    except Exception:nums.append(None)
    pair.append(dict(dataset=domain,sequence_id=sid,position=i,hdf5_index=idx,rgb_path=rgb,gt_path=gt,shared_filename_token=a0,rgb_depth_token_equal=a0==b0,hdf5_co_indexed='YES',official_loader_co_indexed='YES',data_generation_pairing='UNKNOWN',physical_time_pair='UNKNOWN',physical_registration='UNKNOWN'))
    order.append(dict(dataset=domain,sequence_id=sid,position=i,hdf5_index=idx,stored_rgb_filename=Path(rgb).name,stored_depth_filename=Path(gt).name,filename_numeric_token=str(nums[-1]) if nums[-1] is not None else 'UNKNOWN',token_interpretation='filename syntax only; origin UNKNOWN',source_frame_id='UNKNOWN',physical_timestamp='UNKNOWN',ordering='container index + filename token only',temporal_semantics='UNKNOWN'))
   for j in (0,16,31):
    x=r[j]; idx=int(x['hdf5_sequence_index']);node=h[s['depth'][idx]];im=Image.open(io.BytesIO(np.asarray(node).tobytes()));arr=np.asarray(im)
    samples.append(dict(dataset=domain,index=idx,path=s['depth'][idx],png_mode=im.mode,decoded_dtype=str(arr.dtype),shape=str(arr.shape),raw_min=int(arr.min()),raw_max=int(arr.max()),scaled_min=float(arr.min()/1000),scaled_max=float(arr.max()/1000),scaled_units='official evaluation convention: meters; physical calibration unverified'))
   monotonic=all(a is not None and b is not None and b>a for a,b in zip(nums,nums[1:]));no_duplicates=len(set(nums))==len(nums)
   log.append(f'{domain}: 32 historical positions; source filename tokens strictly increasing={monotonic}; distinct={no_duplicates}; original frame ID / timestamp not established; actual depth PNG sampled positions 0,16,31')
   log.append(f'{domain}: shared token matched {sum(v["rgb_depth_token_equal"] for v in pair if v["dataset"]==domain)}/32; decoder samples '+json.dumps([v for v in samples if v['dataset']==domain],ensure_ascii=False))
  csvout('MNV032_RGB_GT_PAIRING.csv',pair);csvout('MNV032_SOURCE_FRAME_ORDER.csv',order)
 except Exception:log.append(domain+' ERROR '+traceback.format_exc())
assert torch.cuda.is_initialized() is False
report('MNV032_OFFICIAL_DATA_PROTOCOL.md','''# 官方 HDF5 评测协议与证据界限

官方 [评测说明](https://github.com/lpiccinelli-eth/Velodepth/blob/main/docs/eval.md) 指向经处理的评测 HDF5 与 `scripts/eval.py`。源码 `scripts/eval.py:98-110,127-143,159-166`：test_mode=True、num_frames=-1、顺序 sampler、validate；`SequenceDataset.load_dataset:77-113` 按 val.txt 顺序关联 HDF5 内 sequence JSON。`get_test_idxs:159-170` 对 num_frames=-1 选择最后最多 32 个连续容器位置；`get_single_sequence:218-243` 使用同一 j 查 `sample['image'][j]` 和 `sample['depth'][j]`。这已确认**官方操作性评测配对**，不能替代生成代码对真实 RGB/GT 同步、深度配准的证明。

HDF5 sequence JSON 是各字段路径列表与逐帧 camera / pose 字段；TUM 为 `sequences.json`、ScanNetVid 为 `sequences_.json`。`utils_decode.py:12-38` 处理深度 PNG（多通道支持 RGB24）并除类的 scale；RGB `decode_rgb` 使用 torchvision 解码。TUM / ScanNetVid 类的 depth_scale=1000，max_depth=10。两序列 32 个同索引路径与文件名 token 见 CSV；不将文件名视为原始时间戳。

未在本轮受限检查的正式评测调用链中找到原始数据转换配对/图像配准生成代码；不等于证明该代码不存在。未重扫数据集、未下载数据、未运行模型。
''')
report('MNV032_PIXEL_GEOMETRY.md','''# 空间坐标链

`ContextCrop`（`velodepth/datasets/pipelines/transforms.py:1624-1694,1731-1866`）测试期 keep_original=True：根据输入计算居中 crop/pad 和网络尺寸，RGB 双三次插值，原 GT 原尺寸保留，保存原 camera 并对 camera 执行 crop/resize；仅 RGB validity_mask 跟随变换。此前两域观察到 RGB 476×630、GT 480×640，此处复用，不等于说二者本身不对齐。

官方 `velodepth/models/velodepth.py:350-378` `forward_test` 对输出 3D points 用 `match_gt(points_pred, depth_gt, padding1=inputs['paddings'], padding2=None)` 回到 GT canvas，取 z 作为预测。`velodepth/utils/misc.py:584-644` 默认去输入 padding、双线性插值至目标尺寸；`BaseDataset.prepare_depth_eval:166-177` 使用 GT 和其 depth_mask。这个流程表明评测代码的输出坐标约定；它不独立证明 HDF5 生成时物理 RGB-D 标定、校正、同步或最终每个像素真实对应。

M05 `gt2_fixed_clip_repair_train.py:225-249` 归一化、padding 后双线性下采样，`make_z_depth` 调用 `velodepth/models/velodepth.py:144-155` 先插值回 padded canvas 再去 padding。MNV03.1 使用 CPU 零张量已证明检查样本 434×574 返回 480×640 的尺寸链；合成尺寸不证明物理配准，也没有测试真实模型预测。

**状态**：官方评测处理的坐标映射已确认；PIXEL_GEOMETRY_CONFIRMED（包括源 RGB-D 真实逐像素配准）为 UNKNOWN/NO-GO。需要维护者提供生成脚本、颜色与深度的标定、RGB-D registration 方法及代表性验证资料。
''')
report('MNV032_DEPTH_UNIT_REPORT.md','''# 官方深度评测尺度与 mask

数据类 `velodepth/datasets/tum.py:5-12` 与 `scannet.py:53-60` 的 TUM/ScanNetVid 均定义 depth_scale=1000、max_depth=10；`utils_decode.py:12-38` 读取 PNG 原始值转 float32 并除以 1000。模型 `velodepth/models/velodepth.py:374` 将使用深度的 RMSE 称为 `MSE_meter`：据此确认 **OFFICIAL_EVALUATION_CONVENTION_CONFIRMED**，即官方评测将解码值按米制约定使用。并未独立验证原始设备的物理标定或生成时是否做过二次缩放。原 PNG 的 mode、dtype、范围和官方 /1000 输出范围列于执行日志（每域 3 帧），以 PIL 实际读取结果为准，不预断解码一定为 uint16。

`base_dataset.py:73-77` test-mode 使用 AnnotationMask(min_value=0, max_value=10)；`pipelines/formating.py:61-105` 的单通道深度掩码为 0<decoded_depth<10，另与已有同名 mask 相交。`base_dataset.py:166-176,229-239` 使用该 mask。M05 `gt2_fixed_clip_repair_train.py:121-138` 使用 GT>0、GT/预测有限且预测>0；与官方评测 mask 不同，不可静默混用。原始 M05 GT /5000 不适用于已转换的本 HDF5；新协议应按 HDF5 解码 /1000 并显式声明使用何种 mask。
''')
report('MNV032_HDF5_PROTOCOL_SPEC.md','''# MNV_HDF5_EVAL_V1 — proposed, not approved implementation

记录字段：dataset、sequence_id、sequence_position、hdf5_index、source_frame_id（UNKNOWN 允许）、rgb_path、gt_path、physical_timestamp（UNKNOWN 允许）、timestamp_status、depth_scale=1000、depth_unit=official-evaluation-metres、camera_params、valid_mask、pairing_basis、geometry_status。

配对 basis 明确为 `OFFICIAL_LOADER_SAME_INDEX`，不能解释成同步或物理配准已证。顺序 basis 为 `HDF5_JSON_LIST_ORDER`；physical_timestamp 为 null、timestamp_status=UNKNOWN；不得复制 logical_step 到时间字段。输入采用独立版本化 adapter，通过原生解码和经验证的 M05 RGB 预处理输出计算链需要的 img / frame_u8 / static 所需配置，保留原始 GT 和 camera；保证官方 /1000 并声明 mask。

原 M05 read_manifest / load_frame 要求真实文件路径、递增 timestamp 与 RGB-GT Δt；新接口不是原 manifest，也不绕过或篡改原验证。提议增加专用调用入口和审计记录，由研究负责人审批；仅在物理配准、单位和顺序各项证据满足既定门槛后运行。仅逻辑 loader step 可用于设计传播年龄，未知真实时间不能推断秒数。该协议为设计稿，未实现、未测试 run_step、未获准用作正式实验。
''')
report('MNV032_FINAL_DECISION.md','''# 最终判断

| 项目 | TUM | ScanNet |
|---|---|---|
| OFFICIAL_PAIRING_CONFIRMED | PASS：官方同索引约定；物理同步 UNKNOWN | PASS：官方同索引约定；物理同步 UNKNOWN |
| PIXEL_GEOMETRY_CONFIRMED | 官方输出到 GT 网格 PASS；源配准 UNKNOWN | 官方输出到 GT 网格 PASS；源配准 UNKNOWN |
| OFFICIAL_DEPTH_UNIT_CONFIRMED | PASS：官方米制评测约定；独立标定 UNKNOWN | PASS：官方米制评测约定；独立标定 UNKNOWN |
| SOURCE_FRAME_ORDER_CONFIRMED | UNKNOWN：名称数值递增不等于原始帧 ID | UNKNOWN：数字标签递增不等于已证明原始帧 ID |
| TEMPORAL_SEMANTICS_CONFIRMED | UNKNOWN | UNKNOWN |
| M05_HDF5_PROTOCOL_DESIGNED | PASS：仅提案，未实施 | PASS：仅提案，未实施 |
| MNV04_STATIC_EVAL_READY | NO-GO：需源配准及适配器验证 | NO-GO：需源配准及适配器验证 |
| MNV04_TEMPORAL_PILOT_READY | NO-GO | NO-GO |

需向数据集维护者获取：1. 本版 HDF5 生成脚本和转换版本；2. RGB-depth 时间关联和一对一/复用规则；3. 对齐目标相机坐标、畸变/重投影与插值方法；4. 深度单位转换链及原始单位；5. 文件名与原始帧索引/时间戳的映射、序列排序和抽样规则。官方评测约定不等同于独立物理标定或已证明真实时间顺序。两个数据域应分别核验。
''')
report('MNV032_EXECUTION_LOG.txt','\n'.join(log)+f'\nCUDA initialized={torch.cuda.is_initialized()}; models=0; run_step=0; downloads=0.\n')
with zipfile.ZipFile(out/'MNV032_DELIVERY.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(out.glob('MNV032_*')):
  if f.is_file() and f.name!='MNV032_DELIVERY.zip':z.write(f,f.name)
print('\n'.join(log));print('CUDA initialized',torch.cuda.is_initialized())
