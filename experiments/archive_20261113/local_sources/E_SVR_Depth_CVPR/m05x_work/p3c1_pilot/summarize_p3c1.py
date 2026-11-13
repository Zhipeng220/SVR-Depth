import csv, hashlib, json, shutil
from pathlib import Path

root=Path.cwd(); out=root/'docs/m05x/p3c1_pilot'; work=root/'m05x_work/p3c1_pilot'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
progress=json.loads((out/'execution_progress.json').read_text(encoding='utf8')); lock=json.loads((out/'pilot_protocol.json').read_text(encoding='utf8'))
results=[]
for row in progress['completed']:
 p=out/'runs'/row['anchor_id']/'verification/invariant_validation.json'
 if p.exists():results.append(json.loads(p.read_text(encoding='utf8')))
status='PASS' if len(results)==len(lock['anchor_ids']) and all(r['status']=='PASS' for r in results) and not progress['stopped_early'] else 'BLOCKED'
dump(out/'invariant_validation.json',{'status':status,'anchors':results,'stopped_early':progress['stopped_early'],'reason':progress['reason']})
geometries=[]; metrics=[]
for r in results:
 if 'details' not in r or 'geometry' not in r['details']:continue
 g=r['details']['geometry']; a=r['anchor_id']; geom={'anchor_id':a,'dataset':r['dataset'],'sequence':r['sequence'],**{k:v for k,v in g.items() if k!='layers'}}
 for layer in g['layers']:
  for k,v in layer.items():
   if k!='layer':geom['layer'+str(layer['layer'])+'_'+k]=v
 geometries.append(geom)
 for b in ['O','V5','V4M','V4N','S']:
  m=r['metrics_absrel'];n=r['valid_pixels'];metrics.append({'anchor_id':a,'dataset':r['dataset'],'sequence':r['sequence'],'branch':b,'H1_AbsRel':m[b+'_H1'],'H4_AbsRel':m[b+'_H4'],'H1_valid_pixels':n[b+'_H1'],'H4_valid_pixels':n[b+'_H4'],'direction_delta_H4':m['direction_delta_H4']})
for name,rows,defaults in [('candidate_geometry.csv',geometries,['anchor_id']),('paired_future_results.csv',metrics,['anchor_id','branch'])]:
 with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else defaults);writer.writeheader();writer.writerows(rows)
for name in ['capture_p3c1.py','verify_p3c1.py','test_verify_p3c1.py','run_p3c1.py','summarize_p3c1.py']:shutil.copy2(work/name,out/name)
for name in ['verify_p3c1.py','test_verify_p3c1.py']:
 p=out/name;p.write_text(p.read_text(encoding='utf8').replace('parents[2]','parents[3]'),encoding='utf8')
lines=['# P3-C1：同范数不同方向试运行报告','',f'**验收状态：{status}。** 本轮只使用预先固定的三个开发 anchor；没有训练，也没有启动 42-anchor Formal。','', '## 冻结协议与实现','',f"- 冻结 checkpoint SHA256：V4 `{lock['checkpoint_sha256']['V4']}`；V5 `{lock['checkpoint_sha256']['V5']}`。",f"- 底座 SHA256：`{lock['backbone_sha256']}`；decoder `{lock['decoder_sha256']}`；修复头源码 `{lock['repair_source_sha256']}`。",'- S0 是相同父状态在当前帧关闭修复后的传播特征；V5 的 pre-readout 分支产生一次共享 D1。V4/V5 两个头在同一 S0 的同一 GPU 混合精度输入上各生成一次候选，提交发生在当前读出之后。非 feature 状态来自 S0，未来四帧均关闭修复且不刷新关键帧。','- V4-matched 以 GPU 实测的四层拼接 L2 比例缩放；匹配误差上限为 1e-5，方向绝对余弦上限为 0.999。L2 先按 float32 求差，再用 float64 累计。只匹配整体范数，不保证各层相等。','- O、V5-native、V4-matched、V4-native、Sham 五条分支保存无损数组。O 与 V5 端点和 P3-B 的所有 H1–H4 预测要求逐位一致；不可事后放宽。GT 在全部未来传播结束后才加载。','', '## 逐 anchor 结果','', '| Anchor | O H1 / H4 | V5 H1 / H4 | V4-matched H1 / H4 | V4-native H1 / H4 | Sham H1 / H4 | H4 差值 V4M−V5 |','|---|---:|---:|---:|---:|---:|---:|']
if results:
 env=json.loads((out/'runs'/results[0]['anchor_id']/'capture_metadata.json').read_text(encoding='utf8'))
 lines.insert(8,f"- 实际运行环境：{env['device']}；PyTorch {env['torch']}；CUDA {env['cuda']}；两个修复头均采用 CUDA autocast float16。")
for r in results:
 if 'metrics_absrel' not in r:continue
 m=r['metrics_absrel']; pair=lambda b:f"{m[b+'_H1']:.6f} / {m[b+'_H4']:.6f}"
 lines.append(f"| {r['anchor_id']} | {pair('O')} | {pair('V5')} | {pair('V4M')} | {pair('V4N')} | {pair('S')} | {m['direction_delta_H4']:+.6f} |")
lines+=['','| Anchor | V4 原始 L2 | V5 原始 L2 | V4 匹配 L2 | 缩放系数 | 方向余弦 | 相对匹配误差 |','|---|---:|---:|---:|---:|---:|---:|']
for g in geometries:lines.append(f"| {g['anchor_id']} | {g['v4_native_l2']:.6f} | {g['v5_native_l2']:.6f} | {g['v4_matched_l2']:.6f} | {g['scale']:.8f} | {g['v4_v5_cosine']:.6f} | {g['matched_norm_relative_error']:.3g} |")
lines+=['','逐层范数、逐层余弦及 H1/H4 有效像素数分别见 `candidate_geometry.csv` 和 `paired_future_results.csv`。','', '## 不变量与证据边界','']
for r in results:
 bad=[k for k,v in r.get('checks',{}).items() if not v];lines.append(f"- {r['anchor_id']}：{r['status']}；失败项：{', '.join(bad) if bad else '无'}。")
lines+=['','独立检查器重新读取 NPZ，重算范数、逐层差异、余弦、掩码和 AbsRel，并核对 P3-B 原始证据文件哈希及端点。实时存储指针、执行顺序和 GT 不参与推理还需要结合采集源码与运行记录审计；仅凭事后数组不能独立证明这两项。相同 D1 是一次共享读出，而非五次独立前向。','', '三个开发 anchor 只支持局部方向干预观察，不代表跨录像泛化或统计显著性；V4-matched 是缩放后的状态，不是原生 V4。V4-native 同时改变方向与幅度，只作诊断。不得将这些结果当成最终测试。','', '## 实际命令与运行记录','', '```powershell','python -m unittest discover -s m05x_work/p3c1_pilot -p \'test_*.py\' -v','python m05x_work/p3c1_pilot/run_p3c1.py','python m05x_work/p3c1_pilot/summarize_p3c1.py','```','', '每个 anchor 的远端命令、退出码和证据位置记录于 `execution_progress.json`，标准输出与检查器输出保存在各自 `runs/*/capture.log` 和 `verify.log`。',f"停止状态：{progress['stopped_early']}；原因：{progress['reason'] or '无'}。",'', '本轮到此停止，是否进入 Formal 等待研究负责人批准。']
(out/'P3C1_PILOT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
files={str(p.relative_to(out)).replace('\\','/'):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'evidence_manifest.json' and '__pycache__' not in p.parts}
dump(out/'evidence_manifest.json',{'status':status,'files':files,'per_array_indexes':'runs/<anchor>/verification/evidence_manifest.json','excluded_self_digest':True})
print(status,len(results),'anchors')
