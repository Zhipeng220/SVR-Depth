from pathlib import Path
import hashlib, json, csv
root=Path.cwd();out=root/'docs/m05x/p4ab'
def write(name,text):(out/name).write_text(text.strip()+'\n',encoding='utf-8')
rows=list(csv.DictReader((out/'P4A_PAIRED_DATA_AUDIT.csv').open(encoding='utf-8-sig',newline='')))
assert len(rows)==39 and all(r['audit_status']=='PASS' for r in rows)
assert sum(r['provisional_group']=='TUM_DESK_ROOM_PROVISIONAL' for r in rows)==6
assert sum(r['provisional_group']=='BONN_RELATED_PROVISIONAL' for r in rows)==6
assert sum(r['provisional_group']=='SCANNET_ALL_UNKNOWN' for r in rows)==15
write('P4B_SYNTHETIC_TEST_REPORT.md', '''# P4-B 合成数据测试报告

状态：**代码与合成数据测试通过；真实数据拟合 BLOCKED**。

模型 M0 为训练折标签均值；M1 为整体及四层更新范数；M2 额外使用四层 C3。M1/M2 为带截距的 Ridge，固定 alpha=1；只用训练折拟合均值与标准差；零方差缩放为 1。留一组交叉验证使用显式组 ID，输出逐组 MAE、RMSE、Spearman 与 M2 相对 M1 的 MAE 改善率；Bootstrap 的重采样单位是组，固定种子。数据集与序列 ID 不进入预测特征。

执行命令（项目根目录）：

```powershell
python m05x_work/p4ab_1_1/test_synthetic.py
```

结果：9/9 PASS。覆盖训练/测试组隔离、折内标准化、未知组及重复样本拒绝、同序列跨组拒绝、真实数据入口默认拒绝、特征白名单、非有限数、常量列与并列秩、指标和 Bootstrap 重现性。测试只调用程序生成的合成数据，没有读取真实标签进行拟合。

`evaluate` 默认拒绝运行，只有显式 `synthetic_only=True` 且所有记录有合成标记才允许测试；低层数值函数也有显式合成标记参数。此标记是操作保护而非不可绕过的安全边界。若未来获得单独批准，须另行审查数据来源、分组及解除限制的代码；现有 UNKNOWN 分组不能当作 CERTIFIED。

注意：当前实现只接受 V5-native，先按组平均真实值和预测值再计算指标；跨数据域同组拒绝。未来任何拟合协议均须先确认这些口径、固定 Bootstrap 的独立单位与足够数量的物理组。没有用真实数据生成 M0/M1/M2 性能。''')
write('P4AB_DATA_LEAKAGE_AUDIT.md', '''# P4AB 数据泄漏与配对审计

- 使用 P3-C1 Formal 清单，不以 P2-B 的 42 条替代 39 条；三个 Pilot anchor 不进入 Formal。
- 每条 C3、整体及四层 L2、H4 标签均对应同一 anchor 的 V5-native 候选。U_H4 = O_H4 − V5_H4；从历史保存的 O/V5 H4 数值计算，未来 GT 不作为输入特征。未将 V5 C3 配给 V4 分支。
- 从冻结配对表、捕获结果及其独立核验记录、Formal 标签/不变量记录交叉检查 anchor、dataset、sequence、age、候选类型、mask 验证、端点和有效像素数。捕获结果校验记录与 CSV 哈希相符；本轮未重新读取全部 NPZ 计算文件哈希。原配对表构建阶段已检查捕获 NPZ 的 SHA 和四层/整体范数。
- 物理来源尚未认证；不同 sequence ID 不自动视为独立组。分组 JSON 的两组候选组合与 ScanNet 全域未知组仅为保守敏感性方案，不是认证。其余未知关系集中在域内 unknown bucket，不能据此证明这些域相互独立。
- 所有 39 条保留；配对有效 39，缺失/无效 0，认证物理组 0。某一已知 TUM 重复 Depth payload 未出现在 42 个既有 H4 哈希中，但不能据此排除其他潜在泄漏或未记录训练历史。
- 训练折标准化、显式组划分、同组及同序列不得跨折、仅数值特征白名单等均已用合成数据测试。用户提供的暂定组未用于真实标签拟合；拟合仍 BLOCKED。
- 本轮只报告各域 sequence 标签级描述性相关和暂定组合均值；未计算显著性、独立泛化或预测性能，未声称 C3 超出 Norm 的增量价值。C3 的几何 mask 不等同于遮挡/对应可靠性。''')
write('P4AB_PROGRESS_REPORT.md', '''# P4AB-1.1 进度报告

P4-A：完成 39/39 Formal anchor 配对、候选和 H4 标签核验、四层 C3 分域描述及保守组均值。Bonn/TUM/ScanNet 分别为 9/15/15 anchor 和 5/8/8 个 sequence 标签；这些标签不等于已认证的独立物理录像。无缺失或无效样本，没有删除 anchor。

P4-B：M0/M1/M2、组交叉验证、折内标准化、指标与组 Bootstrap 已实现；9 项合成数据测试通过。真实标签拟合、真实性能及 GO/NO-GO 继续 **BLOCKED**，需要研究负责人单独批准探索性拟合协议，并先明确物理分组和泄漏风险。本轮没有用真实标签训练预测器。

结果仅为已有开发数据的描述：TUM 四层相关不同；不同域和不同层的方向不一致。保守分组后每域只有 1–2 个组，不能作可靠组级相关或独立泛化判断。详细数值和限制见 P4A 报告。

执行命令（项目根目录）：

```powershell
python m05x_work/p4ab_1_1/run.py
python m05x_work/p4ab_1_1/test_synthetic.py
python m05x_work/p4ab_1_1/finalize.py
```

复用既有 39 次捕获及独立检查记录，本轮未启动 GPU 捕获、VeloDepth 训练或 P5；未修改 checkpoint、历史 H4 和原始结果。分组是在已有实验和来源审计之后指定，并非结果前的预注册。只读了现有配对结果与检查记录；无新增数据域。''')
files=['P4A_PAIRED_DATA_AUDIT.csv','P4A_ASSOCIATION_REPORT.md','P4B_MODEL_IMPLEMENTATION.py','P4B_SYNTHETIC_TEST_REPORT.md','P4AB_GROUPING_CONFIG.json','P4AB_DATA_LEAKAGE_AUDIT.md','P4AB_PROGRESS_REPORT.md']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
write('SHA256SUMS.txt','\n'.join(digest(out/f)+'  '+f for f in files))
print('Finalized',len(files)+1,'files; grouping counts validated.')
