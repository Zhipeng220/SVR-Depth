import csv,hashlib,json,re,zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();out=root/'outputs/P4A_CrossDomain_Replication_20261110_110108';comp=out/'comparison';audit=out/'audit';comp.mkdir(exist_ok=True);audit.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
pre=json.loads((out/'preflight/P4A_CrossDomain_Preflight.json').read_text(encoding='utf8'));status=json.loads((out/'execution_status.json').read_text(encoding='utf8'));smoke=Path('docs/p4a/P4A_C3_CAPTURE_SMOKE_REPORT.md').read_text(encoding='utf8');bonn=json.loads(re.findall(r'```json\n(.*?)\n```',smoke,re.S)[0]);records=[bonn];valid={}
for ds in ['TUM','ScanNet']:
 p=out/ds/'metrics/result.json';v=out/ds/'audit/independent_validation.json'
 if p.exists() and v.exists():
  r=json.loads(p.read_text(encoding='utf8'));check=json.loads(v.read_text(encoding='utf8'));records.append(r);valid[ds]=check
counts=sum(valid.get(x,{}).get('status')=='PASS' for x in ['TUM','ScanNet']);overall='PASS' if pre['status']=='PASS' and counts==2 and all(x.get('pass') for x in records) else ('PARTIAL PASS' if counts==1 else 'FAIL')
rows=[]
for r in records:
 ds=r.get('dataset','Bonn');row={'dataset':ds,'anchor':r['anchor']}
 for i in range(4):row['Layer'+str(i)]=r['layers'][i]['c3']
 rows.append(row)
with (comp/'C3_cross_domain.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=['dataset','anchor','Layer0','Layer1','Layer2','Layer3']);w.writeheader();w.writerows(rows)
(comp/'C3_cross_domain.json').write_text(json.dumps({'status':overall,'rows':rows,'scope':'one development anchor per dataset; descriptive only'},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
fig,ax=plt.subplots(figsize=(7.2,4.3))
for r in rows:ax.plot(range(4),[r['Layer'+str(i)] for i in range(4)],marker='o',label=r['dataset']+' · '+r['anchor'])
ax.axhline(0,color='gray',linewidth=.7);ax.set(xticks=range(4),xlabel='Feature level',ylabel='Geometric masked cosine (C3)',title='Single-anchor C3: descriptive comparison');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(comp/'C3_comparison.png',dpi=160);plt.close(fig)
lines=['# P4-A 跨域 C3 工程复现','',f'**验收：{overall}。** 固定模型、TUM_00_age4 与 ScanNet_00_age4；每域只有一个开发 anchor，结果仅供描述。','', '## C3 四层对照','', '| 数据集 | Anchor | Layer 0 | Layer 1 | Layer 2 | Layer 3 |','|---|---|---:|---:|---:|---:|']
for r in rows:lines.append('| '+r['dataset']+' | '+r['anchor']+' | '+' | '.join(f"{r['Layer'+str(i)]:+.9f}" if r['Layer'+str(i)] is not None else 'UNDEFINED' for i in range(4))+' |')
lines+=['','Bonn 来源为既有 Smoke 报告中的保存结果，本轮没有重跑 Bonn。C3 逐层仅是一组数值，不能推断统计显著性或未来预测能力。','','## 几何与等价性','']
for r in records:
 ds=r.get('dataset','Bonn');lines.append(f"- {ds}：几何有效像素 `"+', '.join(f"L{x['level']} {x['mask_valid']}/{x['mask_total']}" for x in r['layers'])+f"`；hook 数量 `{r['hook_counts']}`；原函数恢复 `{r['hooks_restored']}`；Baseline/Diagnostic 逐位一致 `{all(v['exact'] for v in r['baseline_vs_diagnostic'].values())}`；历史 D1 逐位一致 `{r['historical_d1_equal']}`。")
lines+=['','TUM 首次独立检查器因 `base_feature0` 与实际 `base_feature_0` 的键名不一致失败；原始错误日志和 JSON 保留。修正检查器后，仅重读同一捕获数组完成核验，TUM 才放行 ScanNet；未重跑 TUM GPU，也未调整模型或容差。','', '## 科学解释与后续判断','']
if len(rows)==3:
 l3=[r['Layer3'] for r in rows];lines.append('- A：三个单独 anchor 的 Layer 3 均为正。该现象是有限开发样本的描述性重复，不能推广到数据域总体。' if all(v>0 for v in l3) else '- A：Layer 3 的正向现象没有在三个选定 anchor 全部复现；见上表。')
 lines.append('- B/C：各层数值及符号见上表；存在逐层和样本间差异。一个域一个 anchor，不能把观察到的差异归因为数据域，也不能作统计推断。')
 ratios={r.get('dataset','Bonn'):[round(x['mask_fraction'],6) for x in r['layers']] for r in records};lines.append('- D：几何有效比例（按层）：`'+json.dumps(ratios,ensure_ascii=False)+'`。比例只代表归一化 grid 的有限/范围判定，不是匹配可靠率。')
lines+=['- 遮挡、动态物体与错误光流仍未解决；C3 是几何 mask 下的候选代理，未证明状态可靠性或未来效用预测能力。','- E：不建议仅凭本次工程通过即批准完整 P4-A Pilot。此前独立录像认证、confidence 语义、事前定量阈值和物理源分组仍待负责人审核。','- 当前研究路线值得作为可检验假设保留；不宣称方法成功或跨域泛化，是否继续由研究负责人决定。','', '## 身份与审计','',f"预检：{pre['status']}。原 Bonn 代码从本会话删除变更记录恢复，完整代码和 C3 片段哈希见 `preflight/`；新脚本保留原 C3 运算片段，只适配数据读取和保存，并新增更严格的逐位验收。每域捕获 NPZ、逐数组哈希、grid/mask、独立复核及日志均位于对应目录。",'','所有数据是开发数据；没有重新运行 Bonn，没有训练或访问未来帧/GT；本轮结束后停止，不启动 P4-B。']
(out/'P4A_CrossDomain_Final_Report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(audit/'audit_report.md').write_text('# 独立审计\n\n- 预检：'+pre['status']+'；整体：'+overall+'。\n- TUM 首次独立检查器键名异常和修复记录保留在 TUM/logs；保留原始捕获，无 GPU 重跑。\n- 两域的独立验证重新读取 NPZ，逐数组哈希、mask、候选、父状态、C3 和 A/B 等价；结果见各自 audit/independent_validation.json。\n- 捕获 NPZ 不包含未来帧或 GT；实时 hook 恢复仍依赖运行记录和源码审计。\n- SHA256SUMS 覆盖交付目录（除自身及最终 ZIP）。\n',encoding='utf8')
zip_path=out/'P4A_CrossDomain_Review.zip';sum_path=audit/'SHA256SUMS';files=[p for p in out.rglob('*') if p.is_file() and p not in [zip_path,sum_path] and '__pycache__' not in p.parts]
sum_path.write_text(''.join(sha(p)+'  '+p.relative_to(out).as_posix()+'\n' for p in sorted(files)),encoding='utf8')
for line in sum_path.read_text(encoding='utf8').splitlines():h,path=line.split('  ',1);assert sha(out/path)==h
files.append(sum_path)
with zipfile.ZipFile(zip_path,'w',allowZip64=True) as z:
 for p in sorted(files):z.write(p,p.relative_to(out).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(zip_path) as z:
 bad=z.testzip()
 if bad:raise RuntimeError('ZIP integrity failure '+bad)
print('FINAL',overall,'files',len(files),'zip_bytes',zip_path.stat().st_size,'zip_sha256',sha(zip_path))
