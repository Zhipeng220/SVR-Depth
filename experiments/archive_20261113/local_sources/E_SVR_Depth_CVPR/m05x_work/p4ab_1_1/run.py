import csv, json, math, hashlib, sys, subprocess
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

root=Path.cwd(); out=root/'docs/m05x/p4ab'; out.mkdir(parents=True,exist_ok=True)
work=root/'m05x_work/p4ab_parallel_20261110'; formal=root/'docs/m05x/p3c1_formal'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,s):p.write_text(s,encoding='utf-8')

conf={'version':'P4AB-1.1','status':'PROVISIONAL_NOT_CERTIFIED','purpose':'Conservative descriptive sensitivity only; no independence, fitting or significance claims.',
 'Bonn':{'BONN_RELATED_PROVISIONAL':['person_tracking2','placing_nonobstructing_box','removing_nonobstructing_box'],'BONN_UNKNOWN_SHARED':'all other Bonn sequences'},
 'TUM':{'TUM_DESK_ROOM_PROVISIONAL':['fr1_desk','fr1_desk2','fr1_room'],'TUM_UNKNOWN_SHARED':'all other TUM sequences'},
 'ScanNet':{'SCANNET_ALL_UNKNOWN':'all eight sequence records together; domain holdout only'},
 'matching':'TUM IDs use last path component, then remove an exact _val/_train suffix; no prefix-only matching',
 'unknowns':'all other unknown relations remain unknown; shared buckets are conservative sensitivity choices, not proven same source',
 'analysis':'within-sequence anchor mean, descriptive domain Spearman, provisional group mean; no significance test; grouping version 1.1 follows prior audits',
 'no_real_label_fitting':True,'not_prospective':'Analysis follows prior experiments and audits; this file is not a pre-outcome registration.'}
write(out/'P4AB_GROUPING_CONFIG.json',json.dumps(conf,ensure_ascii=False,indent=2)+'\n')

data=readcsv(root/'docs/p4ab/P4AB_PAIRED_DATASET.csv'); mf=json.loads((formal/'anchor_manifest.json').read_text(encoding='utf-8'))
expected={r['anchor_id']:r for r in mf['formal_anchors']}; pilot=set(mf['pilot_anchor_ids'])
assert len(expected)==39 and len(data)==39 and len({r['anchor_id'] for r in data})==39 and set(expected)=={r['anchor_id'] for r in data} and not pilot.intersection(expected)
sel={r['anchor_id']:r for r in json.loads((work/'capture_selection.json').read_text(encoding='utf-8'))['records']}
progress={r['anchor_id']:r for r in json.loads((work/'capture_progress.json').read_text(encoding='utf-8'))['records']}
labels={r['anchor_id']:r for r in readcsv(formal/'paired_anchor_results.csv')}; frozen=json.loads((work/'paired_dataset_freeze.json').read_text(encoding='utf-8'))
assert sha(root/'docs/p4ab/P4AB_PAIRED_DATASET.csv')==frozen['frozen_dataset_sha256']
def group(d,s):
 if d=='ScanNet': return 'SCANNET_ALL_UNKNOWN'
 if d=='Bonn':return 'BONN_RELATED_PROVISIONAL' if s in conf['Bonn']['BONN_RELATED_PROVISIONAL'] else 'BONN_UNKNOWN_SHARED'
 base=s.replace('\\','/').rsplit('/',1)[-1].removesuffix('_val').removesuffix('_train')
 return 'TUM_DESK_ROOM_PROVISIONAL' if base in conf['TUM']['TUM_DESK_ROOM_PROVISIONAL'] else 'TUM_UNKNOWN_SHARED'
for r in data:
 aid=r['anchor_id']; e=expected[aid]; p=progress[aid]; s=sel[aid]; lab=labels[aid]
 assert r['dataset']==e['dataset']==s['dataset'] and r['sequence_id']==e['sequence']==s['sequence'] and int(r['age'])==int(e['age'])
 assert p['status']=='PASS' and r['candidate']=='V5-native' and r['matched_evidence']=='PASS' and r['physical_group_status']=='UNKNOWN'
 assert float(r['O_H4_AbsRel'])==float(lab['O_H4_AbsRel']) and float(r['V5_H4_AbsRel'])==float(lab['V5_H4_AbsRel'])
 assert math.isclose(float(r['U_H4']),float(r['O_H4_AbsRel'])-float(r['V5_H4_AbsRel']),abs_tol=1e-15) and int(lab['O_H4_valid_pixels'])==int(lab['V5_H4_valid_pixels'])==int(r['H4_valid_pixels'])
 cp=Path(p['path']); result=json.loads((cp/'metrics/result.json').read_text(encoding='utf-8'))
 chk=(work/'reverified'/aid/'independent_validation.json') if p['source']=='reused' else cp/'audit/independent_validation.json'
 inv=json.loads(chk.read_text(encoding='utf-8')); old=json.loads((formal/'runs'/aid/'verification/invariant_validation.json').read_text(encoding='utf-8'))
 assert inv['status']=='PASS' and old['status']=='PASS' and old['checks']['gt_and_metric_masks_equal'] and old['checks']['prior_endpoints_exact']
 assert result['anchor']==aid and result['pass'] and result['capture_npz_sha256']==r['capture_sha256'] and s['evidence_arrays.npz']==r['historical_npz_sha256']
 assert all(float(r[f'c3_l{i}'])==float(result['layers'][i]['c3']) and math.isfinite(float(r[f'c3_l{i}'])) for i in range(4))
 r['provisional_group']=group(r['dataset'],r['sequence_id']);r['capture_npz_path']=str(cp/'capture/capture.npz');r['capture_verification_path']=str(chk);r['formal_validation_path']=str(formal/'runs'/aid/'verification/invariant_validation.json')
 r['audit_status']='PASS';r['group_certification']='PROVISIONAL/UNKNOWN';r['formal_mask_evidence']='PASS'
with (out/'P4A_PAIRED_DATA_AUDIT.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

ridge=(work/'ridge.py').read_text(encoding='utf-8')
ridge=ridge.replace(" keys=set()\n for r in rows:"," keys=set(); anchor_groups={}; sequence_groups={}\n for r in rows:\n  anchor=r.get('anchor_id'); group=r.get('physical_group_id'); sequence=(r.get('dataset'),r.get('sequence_id')) if r.get('sequence_id') else None\n  if anchor in anchor_groups and anchor_groups[anchor]!=group: raise ValueError('anchor spans groups')\n  anchor_groups[anchor]=group\n  if sequence and sequence in sequence_groups and sequence_groups[sequence]!=group: raise ValueError('sequence spans groups')\n  if sequence: sequence_groups[sequence]=group")
ridge=ridge.replace('def evaluate(rows):\n validate(rows);',"def evaluate(rows, *, synthetic_only=False):\n if synthetic_only is not True or any(r.get('synthetic_marker') != 'SYNTHETIC_TEST_ONLY' for r in rows): raise PermissionError('Real-label fitting is BLOCKED; synthetic tests only')\n validate(rows);")
ridge=ridge.replace('def ridge_predict(train_x,train_y,test_x,alpha=1.0):\n', "def ridge_predict(train_x,train_y,test_x,alpha=1.0, *, synthetic_only=False):\n if synthetic_only is not True: raise PermissionError('Real-label fitting is BLOCKED')\n")
ridge=ridge.replace('features[te,:5])[0]', 'features[te,:5],synthetic_only=True)[0]').replace('features[te])[0]', 'features[te],synthetic_only=True)[0]')
ridge='''\"\"\"P4AB-1.1: synthetic-only validation. Real-data fitting remains BLOCKED.\nDo not call low-level ridge_predict on real labels; it is a numerical primitive.\nPhysical group certification is mandatory and IDs are never inferred from sequence names.\"\"\"\n'''+ridge
write(out/'P4B_MODEL_IMPLEMENTATION.py',ridge)

sys.path.insert(0,str(out));import P4B_MODEL_IMPLEMENTATION as model
seq=defaultdict(list)
for r in data:seq[(r['dataset'],r['sequence_id'])].append(r)
seqrows=[]
for (d,s),items in seq.items():
 row={'dataset':d,'sequence':s,'group':items[0]['provisional_group'],'n':len(items)}
 for k in ['U_H4','global_l2']+[f'c3_l{i}' for i in range(4)]+[f'l{i}_l2' for i in range(4)]:row[k]=float(np.mean([float(x[k]) for x in items]))
 seqrows.append(row)
lines=['# P4-A 配对审计与探索性关联报告','','本报告仅对历史开发数据做描述性分析；39 个 Formal anchor 属于 21 个 sequence 标签，物理录像独立性尚未认证。未拟合预测器、未作显著性或泛化声明。', '', '## 配对与来源','',f'- Formal 清单 39/39；Pilot 三个 age1 未纳入。有效 39，缺失或无效 0；Bonn/TUM/ScanNet = 9/15/15 anchor，sequence 标签 = 5/8/8。', '- 只使用 V5-native：C3、四层和整体更新范数均来自对应 anchor 候选；H4 标签为 O−V5，O/V5 有效像素数一致，既有独立检查器记录 mask 和历史端点通过。', '- 检查既有结果、清单、捕获结果及其记录的 SHA：本轮未重新计算所有 NPZ 的文件 SHA；原冻结配对表构建时已逐文件计算捕获 SHA 并校验候选更新范数。详细路径和哈希在 CSV。', '- C3 是几何有效掩码下的候选代理，不能证明遮挡、光流对应或未来效用；样本不存在视为独立物理录像的认证。', '', '## 分域、四层描述性 Spearman', '', '每条 sequence 标签内先平均 anchor，再对标签计算等级相关；仅描述现有开发序列，不提供 p 值。样本数小且相关来源未认证。正 U 表示提交有利。','', '| 数据域 | sequence 标签数 | C3 L0 | C3 L1 | C3 L2 | C3 L3 | U 均值 |', '|---|---:|---:|---:|---:|---:|---:|']
for d in ['Bonn','TUM','ScanNet']:
 q=[x for x in seqrows if x['dataset']==d]; vals=[model.spearman([x[f'c3_l{i}'] for x in q],[x['U_H4'] for x in q]) for i in range(4)]
 lines.append('| '+d+' | '+str(len(q))+' | '+' | '.join('undefined' if x is None else f'{x:.4f}' for x in vals)+f" | {np.mean([x['U_H4'] for x in q]):+.6f} |")
lines+=['','C3 四层数值范围（逐 sequence 平均）：','','| 域 | L0 min/max | L1 min/max | L2 min/max | L3 min/max |','|---|---|---|---|---|']
for d in ['Bonn','TUM','ScanNet']:
 q=[x for x in seqrows if x['dataset']==d]; ranges=[f"{min(x[f'c3_l{i}'] for x in q):.4f} / {max(x[f'c3_l{i}'] for x in q):.4f}" for i in range(4)]
 lines.append('| '+d+' | '+' | '.join(ranges)+' |')
lines+=['','## 保守分组敏感性','','以下先在 sequence 标签内平均，再在组内等权平均；这些组不是已证明的物理实体。由于每域仅 1–2 个保守组，不计算组级相关及区间。ScanNet 全域单组仅用于整体留出的敏感性方案；不能解释为所有 ScanNet 具有同一真实来源。','','| 数据域 | 暂定组 | sequence 标签数 | anchor 数 | U 组均值 | C3 L0 / L1 / L2 / L3 组均值 |','|---|---|---:|---:|---:|---|']
for d in ['Bonn','TUM','ScanNet']:
 for g in sorted({x['group'] for x in seqrows if x['dataset']==d}):
  q=[x for x in seqrows if x['dataset']==d and x['group']==g]; means=[np.mean([x[f'c3_l{i}'] for x in q]) for i in range(4)]
  lines.append(f"| {d} | {g} | {len(q)} | {sum(x['n'] for x in q)} | {np.mean([x['U_H4'] for x in q]):+.6f} | "+' / '.join(f'{v:.4f}' for v in means)+' |')
lines+=['','ScanNet 整域留出时，仅保留以上 Bonn 和 TUM 各自结果，不把它们合并成跨域总相关。由于组数不足，无法评估是否由单一物理来源驱动，也无法论证 C3 超出 Norm 的独立增量价值。','', '此分组依据既有来源审计和本轮指令制定，属于已有结果后的开发性敏感性分析，而非预注册验证。']
write(out/'P4A_ASSOCIATION_REPORT.md','\n'.join(lines)+'\n')
