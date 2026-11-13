import json,csv,hashlib,collections
from pathlib import Path
p=Path('docs/m05x/p5_norm/phase1_3')
def read(n):return json.loads((p/n).read_text(encoding='utf8'))
def savecsv(n,rows):
 with (p/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
bonn=read('manifest.json');tum=read('tum_manifest.json'); traces=[];audits=[]
for dataset,prefix in [('Bonn',''),('TUM','tum_')]:
 m=read(prefix+'manifest.json');t=read(prefix+'trace.json');a=read(prefix+'branch_audit.json');assert m['stage']=='COMPLETED' and len(t)==56 and len(a)==28
 for e in t:
  for layer in e['layer']:
   expected=layer['S1_sha256'] if e['commit'] else layer['S0_sha256'];assert e['feature_hash_after']['feature'+str(layer['layer'])]==expected
  traces.append({'dataset':dataset,'sequence_id':m['selected_sequence'],'repetition':e['repetition'],'strategy':e['group'],'frame':e['frame'],'commit':e['commit'],'predicted_utility':e['predicted_utility'],'global_l2':e['norm_global'],'parent_hashes_json':json.dumps(e['parent_hashes'],sort_keys=True),'state_after_hashes_json':json.dumps(e['state_hash_after'],sort_keys=True),'layers_json':json.dumps(e['layer'],sort_keys=True),'nonfeature_source':e['nonfeature_source'],'next_parent_source':e['next_parent_source']})
 for e in a:audits.append({'dataset':dataset,**e})
 for group in ['G0','G1','G2','G3']:
  e0=[e for e in t if e['group']==group and e['repetition']==0];e1=[e for e in t if e['group']==group and e['repetition']==1];assert [(e['commit'],e['state_hash_after']) for e in e0]==[(e['commit'],e['state_hash_after']) for e in e1]
  for seq in [e0,e1]:
   for previous,current in zip(seq,seq[1:]):assert previous['state_hash_after']==current['parent_hashes']
 assert all(x['parent_clones_storage_distinct'] and x['mutation_isolated'] for x in a)
 first=[e for e in t if e['repetition']==0 and e['frame']==1];assert len(first)==4 and len({json.dumps(e['parent_hashes'],sort_keys=True) for e in first})==1
 assert len({json.dumps([x['S0_sha256'] for x in e['layer']]) for e in first})==1 and len({json.dumps([x['S1_sha256'] for x in e['layer']]) for e in first})==1
savecsv('P5NORM_RUNTIME_TRACE.csv',traces);savecsv('P5NORM_STATE_BRANCH_AUDIT.csv',audits)
manifest={'checkpoint_path':bonn['checkpoint_path'],'checkpoint_bytes':bonn['checkpoint_bytes'],'checkpoint_sha256':bonn['identities']['checkpoint'],'checkpoint_expected_frozen_sha256':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251','frozen_hash_match':True,'loader':bonn['loader'],'checkpoint_load':bonn['checkpoint_load'],'model_config_sha256':bonn['model_config_sha256'],'backbone_sha256':bonn['identities']['backbone'],'m05_sha256':bonn['identities']['m05'],'decoder_sha256':bonn['identities']['decoder'],'repair_head_sha256':bonn['identities']['head'],'decoder_class':bonn['decoder_class'],'repair_head_class':bonn['head_class'],'torch':bonn['torch'],'cuda':bonn['cuda'],'gpu':bonn['gpu'],'warning':'strict repair-head load succeeded; other backbone loading emitted unexpected classification-head keys; full load audit not independently captured'}
(p/'P5NORM_CHECKPOINT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
report='''# P5-Norm-1.3 真实服务器短帧隔离探针

**结果：V5 冻结文件身份及修复头严格加载通过；Bonn/TUM 的短段 Feature 选择、已枚举状态克隆和重复重建检查通过。完整状态隔离、当前 D1 一致性和序列间 Reset 仍未证明，P5-Norm-2 为 NO-GO。** 这些结论不等同于正式策略实现或完整闭环评估。

## 真实运行与静态证据

服务器实际项目 `/home/svrdepth/SVR-Depth/velodepth` 使用真实 M05 源码，M05、decoder、repair head 和底座哈希与冻结身份一致；V5 `final.pt` 大小 2,084,982 字节，SHA256 `2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251` 与冻结值一致。`gt2.load_model` 创建底座，`repair_head_state_dict` 以 `strict=True` 加载成功；详细配置和身份在 checkpoint manifest。日志显示 ConvNeXt 分类头有 unexpected keys，不能将此简化成整套底座权重完全无警告；没有对全部加载结果实施独立完整核验。

PyTorch 2.6.0+cu124、CUDA 12.4，RTX 4060 Laptop GPU；两域均选冻结 P2-B 清单中各自第一条开发序列：Bonn `moving_nonobstructing_box`、TUM `fr1_360_val`。各取 8 帧、每组 7 个传播步，四组各重新创建模型、DIS、关键帧状态，并重复一轮，共每域 56 个事件；未读取 GT 或计算性能。TUM 原始 metadata 哈希与冻结值一致。每组首步父状态、S0 和 S1 的哈希在同域内一致；各组后续使用各自前一步状态，状态哈希链一致。

## S0/S1 与选择

服务器 M05 的 `run_step` 分别以 off 与 pre_readout 模式从同父状态执行；off 的返回 feature 用作 S0，pre-readout 返回 feature 用作 S1。首个 Bonn 帧四层均为 `[1,512,31,41]`、float32、cuda:0，四层值有限且 S0/S1 不同；各事件的逐层 shape、dtype、hash 和更新 L2 见 trace。G1 选择 S1，G2 选择 S0，G3 根据对应开发 fold 的五维预测严格 >0 选择 S1，否则 S0；每个事件的选中 feature 哈希均与目标一致。G1/G2/G3 的 radius/confidence 来源为各自同父 off 状态，rays 来源为各自 parent；G0 使用 pre-readout 完整状态。

**局限**：S0 用 off 路径推得，未在 pre-readout 内部 Hook 独立证明它等于该次 repair head 的实际输入；也未实际计算并比较 D1、未把 G0 和冻结原始 P-full 对照执行。因此 S0/S1 运行时语义不能整体标 GO。G0 每步额外执行一次 off 前向用于诊断，不能宣称与未经包装的原始系统完全等价。

## 状态隔离与 Reset

每次 run 的各传播步，从自己父状态创建两个 clone；对 feature 四层、radius、confidence、rays 分别检查 Tensor storage 指针不同，并修改一个副本首元素后核验另一副本哈希不变。每域保存首步 28 条（4 组 × 7 字段）指针/突变证据；完整逐步检查由生成器执行。四组串行，每组新建模型、DIS 和帧缓存；没有同时驻留四组并直接做跨组内存地址和逐组突变检查。对 decoder 内部隐藏缓存、全局或库内状态没有完整枚举；因此不能声称所有 Mutable Cache / Hidden / Temporal State 已隔离。

每域同一条短序列每组重复构造新模型和初始状态，两轮对应决策和状态哈希逐步完全相同；这验证**销毁重建后的同序列重放**，不证明复用同一模型对象清理缓存，也未执行 A 序列后原对象 Reset 再运行 B 与新进程 B 比较。尽管完成多次独立初始化及两域测试，广义 Sequence Reset 仍 NO-GO。

## M1 Guard 与失败记录

本地隔离 `m1_loader_probe.py` 对 21 套 JSON、39 条历史 OOF 特征完成身份、参数、训练集合和 scaler 检查，最大差 `1.734723475976807e-18`，小于 `1e-10`；错误 fold、未知序列、缺失文件、错误特征顺序、非有限输入均拒绝。该 Guard 是新探针，不是服务器正式策略加载器；服务器短段只使用对应 Bonn fold_00 与 TUM fold_13，21 套并未全部在服务器现场加载。

TUM 首次执行误选 fold_05，被 held-out 身份断言拒绝，发生在模型加载和 GPU forward 前；记录在 `tum_manifest.json` 的首次记录已由重试结果取代，首次异常说明保留于本报告。核对 P4BE fold 身份后，在新的 `/tmp/svrdepth_p5norm_1_3_probe_tum_retry` 使用 fold_13 重试通过；未通过调整模型或阈值掩盖。短段探针的 server 临时脚本均放在 `/tmp` 下独立目录，没有修改项目源码或 checkpoint。

所有 GPU 检查的哈希是生成器保存的摘要；未保存原始张量以便独立逐元素重算。日志中的 shape/hash 与选中状态可独立解析校验，但指针/突变记录依赖探针自报，无法据此推断未列出的隐藏对象。trace 中 G0/G1/G2 也计算预测用于诊断，但只有 G3 用它控制动作；无 GT。

## 未完成与判定

checkpoint 文件与修复头严格加载：GO（限本探针）。S0/S1 Semantics：NO-GO（pre-readout 内部 S0、共享 D1 未独立捕获）。G1/G2/G3 State Isolation：NO-GO（未做所有可变缓存/隐藏状态及同时跨组无别名测试）。Sequence Reset：NO-GO（未验证复用对象跨序列 reset）。M1 Runtime Loading：NO-GO（21 折只在本地 Guard，服务器只跑两个折；正式加载器未实现）。P5-Norm-2：NO-GO。建议先补充受控 hook 的 D1/S0 同次验证、四组并存存储与隐藏状态检查、同进程 A→Reset→B 对比，以及服务器端 21 折错误加载保护；再由负责人审批。
'''
(p/'P5NORM_RUNTIME_ISOLATION_REPORT.md').write_text(report,encoding='utf8')
decision='''# P5-Norm-1.3 最终决策

| 项目 | 决策 | 证据与限制 |
|---|---|---|
| Checkpoint Loading | GO（限定） | 冻结 SHA 匹配、修复头 strict 加载；底座载入有分类头 unexpected keys，完整基座加载清单未单独验收。 |
| S0/S1 Semantics | NO-GO | 两域短段输出有限且不同，选择值哈希正确；未 Hook 验证 pre-readout 实际 S0 或共享 D1。 |
| G1/G2/G3 State Isolation | NO-GO | 已枚举七个状态字段的 clone 指针/突变通过；组串行重建，未证明所有内部缓存、hidden state 和跨组无别名。 |
| Sequence Reset | NO-GO | 两域各两轮新模型重建完全重现；未验证复用对象跨不同序列 Reset。 |
| M1 Runtime Loading | NO-GO | 本地 21 折保护和 39 预测通过；服务器只加载 2 折，正式服务器 Loader 和错误折拒绝未验收。 |
| P5-Norm-2 | NO-GO | 存在上述关键状态与 Loader 证据缺口。 |

冻结错误 Fold 首次被拒绝的情况、重试与所有限制详见 `P5NORM_RUNTIME_ISOLATION_REPORT.md`。不自动启动下一阶段，等待负责人审批。
'''
(p/'P5NORM_PHASE1_3_DECISION.md').write_text(decision,encoding='utf8')
files=sorted(f for f in p.rglob('*') if f.is_file() and f.name!='SHA256SUMS.txt');(p/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.relative_to(p).as_posix()+'\n' for f in files),encoding='utf8');print('PASS independently parsed',len(traces),'events;',len(audits),'clone-field records;',len(files),'files hashed')
