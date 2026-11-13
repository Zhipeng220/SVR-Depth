import csv,hashlib,json,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
root=Path.cwd();out=root/'docs/m05x/p3c1_formal';work=root/'m05x_work/p3c1_formal'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(name,x):(out/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def csvout(name,rows,default):
 with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else default);w.writeheader();w.writerows(rows)
manifest=json.loads((out/'anchor_manifest.json').read_text(encoding='utf8'));progress=json.loads((out/'execution_progress.json').read_text(encoding='utf8'));results=[];bad=[]
for p in progress['completed']:
 vd=out/'runs'/p['anchor_id']/'verification/invariant_validation.json'
 if not vd.exists():bad.append(p);continue
 r=json.loads(vd.read_text(encoding='utf8'))
 if r['status']=='PASS':results.append(r)
 else:bad.append(r)
complete=len(results)==39 and not progress['stopped_early'] and not bad
dump('invariant_validation.json',{'status':'PASS' if complete else 'BLOCKED','expected':39,'valid':len(results),'invalid_or_blocked':bad,'stopped_early':progress['stopped_early'],'reason':progress['reason'],'anchors':results})
rows=[];geometry=[]
for r in results:
 m=r['metrics_absrel'];n=r['valid_pixels'];g=r['details']['geometry'];row={'anchor_id':r['anchor_id'],'dataset':r['dataset'],'sequence':r['sequence']}
 for b in ['O','V5','V4M','V4L','V4N','S']:
  for h in [1,4]:row[f'{b}_H{h}_AbsRel']=m[f'{b}_H{h}'];row[f'{b}_H{h}_valid_pixels']=n[f'{b}_H{h}']
 for h in [1,4]:
  for b in ['V5','V4M','V4L','V4N']:row[f'{b}_minus_O_H{h}']=m[f'{b}_H{h}']-m[f'O_H{h}']
  row[f'global_minus_v5_H{h}']=m[f'direction_delta_H{h}'];row[f'layerwise_minus_v5_H{h}']=m[f'layerwise_delta_H{h}'];row[f'layerwise_minus_global_H{h}']=m[f'V4L_H{h}']-m[f'V4M_H{h}']
 rows.append(row);gr={'anchor_id':r['anchor_id'],'dataset':r['dataset'],'sequence':r['sequence'],**{k:v for k,v in g.items() if k!='layers'}}
 for l in g['layers']:
  for k,v in l.items():
   if k!='layer':gr[f"layer{l['layer']}_{k}"]=v
 geometry.append(gr)
csvout('paired_anchor_results.csv',rows,['anchor_id']);csvout('global_vs_layerwise_geometry.csv',geometry,['anchor_id'])
groups=defaultdict(list)
for r in rows:groups[(r['dataset'],r['sequence'])].append(r)
seq=[];cols=[k for k in rows[0] if k not in ['anchor_id','dataset','sequence'] and 'valid_pixels' not in k] if rows else []
for (ds,name),rs in sorted(groups.items()):
 sr={'dataset':ds,'sequence':name,'n_anchors':len(rs),'anchor_ids':';'.join(x['anchor_id'] for x in rs)}
 for k in cols:sr[k]=sum(x[k] for x in rs)/len(rs)
 seq.append(sr)
csvout('sequence_level_results.csv',seq,['dataset','sequence'])
summaries={};intervals={};rng=np.random.default_rng(manifest['bootstrap']['seed']);nboot=manifest['bootstrap']['resamples']
for ds in ['Bonn','TUM','ScanNet']:
 sr=[x for x in seq if x['dataset']==ds];summaries[ds]={'n_sequences':len(sr),'n_anchors':sum(x['n_anchors'] for x in sr),'endpoints':{}};intervals[ds]={}
 for col in ['layerwise_minus_v5_H4','global_minus_v5_H4','layerwise_minus_global_H4','layerwise_minus_v5_H1','global_minus_v5_H1']:
  if not sr:continue
  x=np.array([r[col] for r in sr]);means=x[rng.integers(0,len(x),size=(nboot,len(x)))].mean(axis=1);lo,hi=np.quantile(means,[.025,.975]);h1col=col.replace('H4','H1')
  summaries[ds]['endpoints'][col]={'mean':float(x.mean()),'median':float(np.median(x)),'positive_count':int((x>0).sum()),'negative_count':int((x<0).sum()),'zero_count':int((x==0).sum()),'min':float(x.min()),'max':float(x.max()),'h1_h4_same_sign_count':int(sum(np.sign(r[col])==np.sign(r[h1col]) for r in sr)) if h1col!=col else None}
  intervals[ds][col]={'lower_95':float(lo),'upper_95':float(hi),'bootstrap_unit':'sequence','resamples':nboot,'seed':manifest['bootstrap']['seed']}
dump('dataset_summary.json',{'status':'complete' if complete else 'partial_not_formal','datasets':summaries,'sequence_weight':'equal after averaging available nonpilot anchors within sequence'})
dump('confidence_intervals.json',{'status':'complete' if complete else 'partial_not_formal','datasets':intervals,'interpretation':'exploratory percentile intervals over selected development sequences, not independent generalization'})
for name in ['capture_formal.py','verify_formal.py','test_verify_formal.py','run_formal.py','prepare_formal.py','summarize_formal.py','freeze_and_audit.py']:
 shutil.copy2(work/name,out/name)
for name in ['verify_formal.py','test_verify_formal.py']:
 p=out/name;p.write_text(p.read_text(encoding='utf8').replace('parents[2]','parents[3]'),encoding='utf8')
lines=['# P3-C1 Formal 同范数方向实验报告','',f"验收：**{'PASS' if complete else 'BLOCKED / 部分结果'}**。有效 {len(results)}/39，失败或缺失 {39-len(results)}。",'','## 方案与边界','','预先排除三个 Pilot anchor，其余 39 个原 P2-B 开发 anchor 固定入组；同一录像内先平均，再等权统计录像。原冻结 V4/V5 和底座只用于推理；D1 一次共享，未来 H1–H4 关闭修复，不刷新关键帧。逐层及整体匹配均使用 1e-5 相对容差，端点和 sham 逐位一致。','', '逐层匹配控制每一层的更新 L2，仍不保证层内位置、通道及方向分布相同。V4 native 同时改变幅度与方向；不能作为单独方向效应。所有结果均为已接触的开发录像，不构成独立泛化验证。','', '## 按录像汇总（H4）','', '| 数据集 | 录像数 | 逐层匹配−V5 均值 | 中位数 | 95% Bootstrap 区间 | 正 / 负 | 全局匹配−V5 均值 |', '|---|---:|---:|---:|---:|---:|---:|']
for ds in ['Bonn','TUM','ScanNet']:
 d=summaries[ds];p=d['endpoints'].get('layerwise_minus_v5_H4');q=d['endpoints'].get('global_minus_v5_H4');ci=intervals[ds].get('layerwise_minus_v5_H4')
 if p:lines.append(f"| {ds} | {d['n_sequences']} | {p['mean']:+.7f} | {p['median']:+.7f} | [{ci['lower_95']:+.7f}, {ci['upper_95']:+.7f}] | {p['positive_count']} / {p['negative_count']} | {q['mean']:+.7f} |")
lines+=['','区间以录像为重采样单位，固定随机种子及 10,000 次重采样；样本量小，区间仅作开发数据的探索性描述。每录像 anchor 数可能不同，因为三个 Pilot anchor 不进入主分析。','', '## 核验与可复现性','',f"- 有效 anchor：{len(results)}；冻结清单、逐项检查、远端命令和日志分别见 `anchor_manifest.json`、`invariant_validation.json`、`execution_progress.json` 和 `runs/`。",'- 独立检查器读取无损数组，检查六分支状态、逐层匹配、保存状态与计算公式、P3-B 端点、输入与缓存、掩码和 sham；采集日志记录每条 replay 完成与 GT 加载开始的顺序。实时指针及执行顺序仍需结合程序和运行记录审计，事后数组不能独立证明真实执行过程。','- Bootstrap、逐录像结果与逐层几何分别见 `confidence_intervals.json`、`sequence_level_results.csv`、`global_vs_layerwise_geometry.csv`。','', '## 研究问题与停止结论','']
if complete:
 lines+=['- A/B：整体及逐层匹配的配对差值见上表和逐录像数据；这些比较支持或限制的是本次固定候选方向的局部因果效应。','- C：逐层与整体结果的差异涉及跨层更新能量分配，但两种人为缩放同时改变完整向量，不能把效应唯一归因于跨层能量。','- D：录像级正负计数和范围见 `dataset_summary.json`，不得把 anchor 当独立录像。','- E：是否进入 Future-Utility-Aware 方法开发需结合效应稳定性和独立数据隔离由研究负责人决策；当前未认证独立测试数据，不可声称泛化。']
else:lines+=['全部正式 anchor 尚未完成；A–E 无法据此作正式结论。保留现有运行记录并排查停止原因。']
lines+=['','执行命令：`python m05x_work/p3c1_formal/freeze_and_audit.py`、`python m05x_work/p3c1_formal/prepare_formal.py`、`python -m unittest discover -s m05x_work/p3c1_formal -p test_*.py -v`、`python m05x_work/p3c1_formal/run_formal.py`、`python m05x_work/p3c1_formal/summarize_formal.py`。','', '本轮无训练，不启动 P4；完成后停止等待审核。']
(out/'P3C1_FORMAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
files={str(p.relative_to(out)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'evidence_manifest.json' and '__pycache__' not in p.parts}
dump('evidence_manifest.json',{'status':'PASS' if complete else 'BLOCKED','files':files,'array_indexes':'runs/<anchor>/verification/evidence_manifest.json','excluded_self_digest':True})
print('SUMMARY', 'PASS' if complete else 'BLOCKED',len(results),'of 39')
