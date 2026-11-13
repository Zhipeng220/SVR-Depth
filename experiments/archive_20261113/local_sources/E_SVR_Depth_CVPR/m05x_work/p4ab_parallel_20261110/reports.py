import csv,json,sys
from pathlib import Path
root=Path.cwd();w=root/'m05x_work/p4ab_parallel_20261110';out=root/'docs/p4ab';sys.path.insert(0,str(w));import ridge,association
state=json.loads((w/'paired_dataset_freeze.json').read_text(encoding='utf8'));lock=json.loads((w/'protocol_lock.json').read_text(encoding='utf8'))
with (out/'P4AB_PAIRED_DATASET.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
group_ok=bool(rows) and state['unknown_physical_rows']==0 and state['physical_groups_certified']>=3 and state['complete'];a_result=None;p_result=None;block_reason=''
if group_ok:
 try:
  a_result=association.analyze(rows,seed=lock['seed'],permutations=lock['permutations']);ev=ridge.evaluate(rows);allg=ev['groups'];per={d:[g for g in allg if g['dataset']==d] for d in ['Bonn','TUM','ScanNet']}
  if any(len(g)<2 for g in per.values()):raise ValueError('fewer than two groups in a domain')
  overall=ridge.aggregate(allg);domain={d:ridge.aggregate(g) for d,g in per.items()};ci=ridge.bootstrap_delta(allg,seed=lock['seed'],resamples=lock['bootstrap_resamples']);dcis={d:ridge.bootstrap_delta(g,seed=lock['seed'],resamples=lock['bootstrap_resamples']) for d,g in per.items()}
  loo=[ridge.aggregate([g for g in allg if g['physical_group_id']!=held])['mae_improvement_rate'] for held in sorted({g['physical_group_id'] for g in allg})];dchange=[domain[d]['M1']['MAE']-domain[d]['M2']['MAE'] for d in per]
  pgo=overall['mae_improvement_rate'] is not None and overall['mae_improvement_rate']>0 and ci['lower']>0 and all(x>=0 for x in dchange) and sum(x>0 for x in dchange)>=2 and all(x is not None and x>0 for x in loo)
  p_result={'overall':overall,'domain':domain,'group_bootstrap_95':ci,'domain_bootstrap_95':dcis,'drop_one_group_improvements':loo,'decision':'GO' if pgo else 'NO-GO','group_predictions':allg,'split_count':len(ev['splits'])}
 except Exception as e:block_reason=repr(e);a_result=None;p_result=None
else:block_reason='物理录像分组未全部认证，或配对样本缺失；严格按冻结协议不运行组统计和真实样本训练。'
if a_result:
 supported=[d for d in ['Bonn','TUM','ScanNet'] if a_result[d].get('status')=='EXPLORATORY' and any(x['partial_permutation_p_holm'] is not None and x['partial_permutation_p_holm']<.05 for x in a_result[d]['c3'])]
 ago='GO' if len(supported)>=2 else 'NO-GO'
else:ago='BLOCKED'
counts=f"有效 {state['n_valid']}/39 anchor；缺失或无效 {state['n_invalid']}；Bonn/TUM/ScanNet 分别 {state['domain_anchors']['Bonn']}/{state['domain_anchors']['TUM']}/{state['domain_anchors']['ScanNet']}。开发序列记录 {state['sequence_records']}；已认证物理录像组 {state['physical_groups_certified']}，独立录像总数 UNKNOWN；未认证组的样本 {state['unknown_physical_rows']}。"
common='\n\n'+counts+'\n\n配对表 SHA256 '+state['frozen_dataset_sha256']+'；协议 SHA256 '+state['protocol_lock_sha256']+'。样本仅 V5-native，与同父 O 匹配；标签是 O−V5 的 H4 AbsRel。所有录像均为开发数据。'
alines=['# P4-A：C3 与未来效用关联',f'决策：**{ago}**。'+common]
if a_result:alines+=['## 分域关联与 Holm 结果',json.dumps(a_result,ensure_ascii=False,indent=2),'P 值为物理录像组置换，四层按域和分析族分别 Holm；少于 8 组时五个范数控制的偏相关不计算。']
else:alines+=['## 未执行的统计',block_reason,'没有把 anchor 或未认证的 sequence ID 当成独立物理录像。四层 Spearman、控制范数后的关联、Holm 和按利害分组分布均为 **未计算**；因此不能判断 C3 的独立增量价值或跨域一致性。']
alines+=['## 限制','几何 mask 不证明无遮挡或光流正确；因果效用标签来自历史同父干预，但跨录像相关本身不能证明兼容性机制或预测能力。']
(out/'P4A_ASSOCIATION_REPORT.md').write_text('\n\n'.join(alines)+'\n',encoding='utf8')
plines=['# P4-B：Future-Utility Ridge 预测',f"决策：**{p_result['decision'] if p_result else 'BLOCKED'}**。"+common]
if p_result:plines+=['## LOSO 与录像级 Bootstrap',json.dumps(p_result,ensure_ascii=False,indent=2),'M0 为训练折均值；M1 为五个 norm；M2 为 norm 加四个 C3。固定 alpha=1；每折标准化只使用训练组。样本量小，只能视作开发评估。']
else:plines+=['## 真实样本训练状态',block_reason,'**未对任何真实样本拟合** M0/M1/M2；MAE、RMSE、Spearman、M2 相对 M1 的改善和录像级区间均不可判定。','## 管线准备','已完成 7 项合成单元测试，涵盖同组隔离、折内标准化、未知物理源拒绝、重复键、Bootstrap 可重现、Holm 和秩次边界；合成测试不是预测性能证据。']
(out/'P4B_PREDICTOR_REPORT.md').write_text('\n\n'.join(plines)+'\n',encoding='utf8')
decision=['# P4AB 最终决策',common,f"- P4-A：**{ago}**。",f"- P4-B：**{p_result['decision'] if p_result else 'BLOCKED'}**。",f"- M2 是否优于 M1：{'见预测报告和冻结准则' if p_result else '无法判断；未拟合真实数据'}。",f"- C3 独立增量价值：{'见关联报告' if a_result else '无法判断；分组条件未满足'}。",'- 无独立盲测结论；不进入 P5，不训练复杂网络。','下一阶段需要能够核查的物理源/时间重叠映射；在用户确认并冻结分组后，才可对共享样本执行录像级评估。','无效或缺失条目：',json.dumps(state['invalid'],ensure_ascii=False,indent=2)]
(out/'P4AB_FINAL_DECISION.md').write_text('\n'.join(decision)+'\n',encoding='utf8')
print('REPORTS',ago,p_result['decision'] if p_result else 'BLOCKED',state['n_valid'])

