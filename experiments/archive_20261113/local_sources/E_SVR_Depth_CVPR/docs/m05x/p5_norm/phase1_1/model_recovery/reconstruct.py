import csv,json,hashlib,math
from pathlib import Path
import numpy as np
root=Path.cwd();out=root/'docs/m05x/p5_norm/phase1_1/model_recovery';data=root/'docs/m05x/p4ab/P4A_PAIRED_DATA_AUDIT.csv';base=root/'docs/m05x/p4ab/p4b_sequence_exploratory_fit';cfg=base/'P4BE_CONFIG.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
config=json.loads(cfg.read_text(encoding='utf8'));assert sha(data)==config['data_sha256'];assert config['ridge_alpha']==1.0
names=['global_l2','l0_l2','l1_l2','l2_l2','l3_l2'];assert config['models']['M1']==names
lock={'tolerance_max_abs':1e-10,'feature_names':names,'source_data_hash':sha(data),'model_config_hash':sha(cfg),'formula':'training mean, population std; zero std=1; ridge alpha=1, unpenalized intercept','expected_folds':21,'label':'U_H4=O_H4-V5_H4'}
write(out/'P5NORM_RECOVERY_LOCK.json',lock)
rows=read(data);old=read(base/'P4BE_OOF_PREDICTIONS.csv');lookup={r['anchor_id']:r for r in old};assert len(rows)==39 and len(lookup)==39 and set(lookup)=={r['anchor_id'] for r in rows}
groups=sorted({(r['dataset'],r['sequence_id']) for r in rows});assert len(groups)==21
x=np.array([[float(r[k]) for k in names] for r in rows]);y=np.array([float(r['U_H4']) for r in rows]);assert np.isfinite(x).all() and np.isfinite(y).all()
assert all(abs(float(r['U_H4'])-(float(r['O_H4_AbsRel'])-float(r['V5_H4_AbsRel'])))<1e-15 for r in rows)
result=[];checks=[]
for fold,(dataset,seq) in enumerate(groups):
 test=np.array([(r['dataset'],r['sequence_id'])==(dataset,seq) for r in rows]);train=~test
 trainids=sorted({(rows[i]['dataset'],rows[i]['sequence_id']) for i in np.flatnonzero(train)});assert len(trainids)==20 and (dataset,seq) not in trainids
 mu=x[train].mean(0);scale=x[train].std(0);scale=np.where(scale==0,1,scale);z=(x[train]-mu)/scale;intercept=y[train].mean();coef=np.linalg.solve(z.T@z+np.eye(5),z.T@(y[train]-intercept));pred=intercept+((x[test]-mu)/scale)@coef
 for i,v in zip(np.flatnonzero(test),pred):
  r=rows[i];oldp=float(lookup[r['anchor_id']]['M1_prediction']);assert (r['dataset'],r['sequence_id'])==(lookup[r['anchor_id']]['dataset'],lookup[r['anchor_id']]['sequence_id']);diff=abs(float(v)-oldp);assert diff<=1e-10
  result.append({'fold_id':fold,'anchor_id':r['anchor_id'],'dataset':dataset,'sequence_id':seq,'U_H4':float(y[i]),'reconstructed_prediction':float(v),'original_prediction':oldp,'absolute_difference':diff})
 write(out/f'fold_{fold:02d}.json',{'fold_id':fold,'heldout_dataset':dataset,'heldout_sequence_id':seq,'train_sequence_ids':[{'dataset':a,'sequence_id':b} for a,b in trainids],'train_anchor_ids':[rows[i]['anchor_id'] for i in np.flatnonzero(train)],'heldout_anchor_ids':[rows[i]['anchor_id'] for i in np.flatnonzero(test)],'feature_names':names,'scaler_mean':mu.tolist(),'scaler_scale':scale.tolist(),'ridge_coef':coef.tolist(),'ridge_intercept':float(intercept),'alpha':1.0,'training_sample_count':int(train.sum()),'source_data_hash':sha(data),'model_config_hash':sha(cfg)})
 checks.append({'fold':fold,'n':int(test.sum()),'max':max(r['absolute_difference'] for r in result if r['fold_id']==fold)})
with (out/'P5NORM_M1_RECONSTRUCTED_PREDICTIONS.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
# Independent re-read saved JSON and recompute by raw-space coefficients.
for r in result:
 m=json.loads((out/f"fold_{r['fold_id']:02d}.json").read_text(encoding='utf8'));v=di=next(a for a in rows if a['anchor_id']==r['anchor_id']);coef=np.array(m['ridge_coef']);sc=np.array(m['scaler_scale']);mu=np.array(m['scaler_mean']);p=float(m['ridge_intercept'])-float((coef*mu/sc).sum())+sum(float(v[k])*float(coef[i]/sc[i]) for i,k in enumerate(names));assert abs(p-r['original_prediction'])<=1e-10
diffs=[r['absolute_difference'] for r in result];err=np.array([r['reconstructed_prediction']-r['U_H4'] for r in result]);olderr=np.array([r['original_prediction']-r['U_H4'] for r in result]);mae_diff=abs(np.mean(abs(err))-np.mean(abs(olderr)));rmse_diff=abs(np.sqrt(np.mean(err**2))-np.sqrt(np.mean(olderr**2)))
seqerr=np.array([np.mean([r['reconstructed_prediction']-r['U_H4'] for r in result if r['fold_id']==i]) for i in range(21)]);seqold=np.array([np.mean([r['original_prediction']-r['U_H4'] for r in result if r['fold_id']==i]) for i in range(21)]);seqmae=abs(np.mean(abs(seqerr))-np.mean(abs(seqold)));seqrmse=abs(np.sqrt(np.mean(seqerr**2))-np.sqrt(np.mean(seqold**2)))
lines=['# 21 折 M1 模型恢复报告','','状态：PASS；只用于相应开发 sequence 留出实验，物理独立性未认证。未训练 VeloDepth 或使用 GPU。',f"预冻结容差 1e-10；39 条预测 / 21 折，最大绝对差 {max(diffs):.17g}；anchor 等权 MAE 差 {mae_diff:.17g}，RMSE 差 {rmse_diff:.17g}；sequence 等权 MAE 差 {seqmae:.17g}，RMSE 差 {seqrmse:.17g}。",'每折训练 20 条 sequence，测试完整留出；检查训练和测试组无交集，特征顺序与原配置一致。scaler 训练折总体标准差，零方差设 1，Ridge α=1，截距不惩罚。保存的 JSON 经独立重读，以原始坐标系代数公式复算所有 OOF 预测。','输入、配置哈希及容差记录在 P5NORM_RECOVERY_LOCK.json。','', '| 折 | 测试 anchor 数 | 最大绝对差 | 状态 |','|---:|---:|---:|---|']
for c in checks:lines.append(f"| {c['fold']} | {c['n']} | {c['max']:.17g} | PASS |")
(out/'P5NORM_M1_MODEL_RECOVERY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('PASS',len(checks),len(result),max(diffs),mae_diff,rmse_diff)
