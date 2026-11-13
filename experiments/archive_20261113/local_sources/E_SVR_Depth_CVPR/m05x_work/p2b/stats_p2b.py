"""Sequence-clustered summaries for frozen P2-B, no model inference."""
import csv,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

def summarize(rows,expected,seed=20261025,reps=10000):
    groups=defaultdict(list);seen=set()
    for r in rows:
        key=(r['dataset'],r['sequence'],int(r['age']))
        if key in seen:raise ValueError(f'duplicate anchor {key}')
        seen.add(key)
        if r['status']!='PASS':raise ValueError(f'invalid anchor {key}')
        groups[(key[0],key[1])].append(r)
    if seen!=expected:raise ValueError(f'anchor coverage mismatch missing={len(expected-seen)} unexpected={len(seen-expected)}')
    seq=[]
    for (dataset,sequence),vals in sorted(groups.items()):
        if sorted(int(r['age']) for r in vals)!=[1,4]:raise ValueError('expected ages 1 and 4')
        entry={'dataset':dataset,'sequence':sequence,'n_anchors':len(vals),'H1_T_minus_O':statistics.mean(float(r['T_minus_O_H1']) for r in vals),'H4_T_minus_O':statistics.mean(float(r['T_minus_O_H4']) for r in vals),'pilot_p2a_exposed':bool(vals[0]['pilot_p2a_exposed']),'m04_exposed':bool(vals[0].get('m04_exposed',False))}
        if all('feature_update_norms' in v for v in vals):
            norms=[json.loads(v['feature_update_norms']) for v in vals]
            entry['feature_norms_equal_anchor_mean']=json.dumps({layer:{metric:statistics.mean(n[layer][metric] for n in norms) if all(n[layer][metric] is not None for n in norms) else None for metric in ('delta_l2','relative_l2')} for layer in norms[0]})
            entry['nonfeature_fields_unchanged']=all(v['nonfeature_fields_unchanged'] for v in vals);entry['keyframe_refresh_during_replay']=False
        seq.append(entry)
    summary={};ci={}
    for dataset in sorted({x['dataset'] for x in seq}):
        subset=[x for x in seq if x['dataset']==dataset];n=len(subset)
        if n<2:raise ValueError(f'{dataset}: insufficient independent recordings for CI')
        d={'n_sequences':n,'n_anchors':2*n,'p2a_exposed_count':sum(x['pilot_p2a_exposed'] for x in subset),'m04_exposed_count':sum(x['m04_exposed'] for x in subset),'h1_h4_same_sign_proportion':sum(np.sign(x['H1_T_minus_O'])==np.sign(x['H4_T_minus_O']) for x in subset)/n}
        for horizon in ('H1','H4'):
            values=np.array([x[horizon+'_T_minus_O'] for x in subset],dtype=np.float64);rng=np.random.default_rng(seed)
            boot=values[rng.integers(0,n,size=(reps,n))].mean(axis=1);interval=np.quantile(boot,[.025,.975])
            d[horizon]={'sequence_mean':float(values.mean()),'sequence_median':float(np.median(values)),'sequence_std_ddof1':float(values.std(ddof=1)),'min':float(values.min()),'max':float(values.max()),'positive_proportion':float((values>0).mean()),'negative_proportion':float((values<0).mean()),'zero_proportion':float((values==0).mean())}
            unseen=np.array([x[horizon+'_T_minus_O'] for x in subset if not x['pilot_p2a_exposed']],dtype=float)
            d[horizon]['not_p2a_exposed_n']=int(len(unseen));d[horizon]['not_p2a_exposed_mean']=float(unseen.mean()) if len(unseen) else None
            unseen_m04=np.array([x[horizon+'_T_minus_O'] for x in subset if not x['m04_exposed']],dtype=float)
            d[horizon]['not_m04_exposed_n']=int(len(unseen_m04));d[horizon]['not_m04_exposed_mean']=float(unseen_m04.mean()) if len(unseen_m04) else None
            ci.setdefault(dataset,{})[horizon]={'lower':float(interval[0]),'upper':float(interval[1]),'method':'sequence bootstrap percentile','n_sequences':n,'replicates':reps,'seed':seed}
        summary[dataset]=d
    return seq,summary,ci

def main():
 root=Path(__file__).resolve().parents[2];out=root/'docs/m05x/p2b';anchor=json.loads((out/'anchor_manifest.json').read_text(encoding='utf-8'))['anchors'];rows=[]
 for item in anchor:
  path=out/'runs'/item['anchor_id']/'verification/invariant_validation.json'
  if not path.is_file():raise ValueError(f'missing verification {item["anchor_id"]}')
  result=json.loads(path.read_text(encoding='utf-8'))
  if (result.get('dataset'),result.get('sequence'),result.get('age'))!=(item['dataset'],item['sequence'],item['age']):raise ValueError(f'anchor identity mismatch: {item["anchor_id"]}')
  rows.append({**item,**result['metrics_absrel'],'status':result['status'],'feature_update_norms':json.dumps(result['details']['feature_update_norms']),'valid_pixels':json.dumps(result['valid_pixels']),'nonfeature_fields_unchanged':bool(result['checks']['nonfeature_state_unchanged'])})
 expected={(a['dataset'],a['sequence'],a['age']) for a in anchor};seq,summary,ci=summarize(rows,expected)
 for name,items in [('paired_anchor_results.csv',rows),('sequence_level_results.csv',seq)]:
  fields=list(dict.fromkeys(k for r in items for k in r))
  with (out/name).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(items)
 for name,obj in [('dataset_summary.json',summary),('confidence_intervals.json',ci)]: (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('DATASET_SUMMARY',json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
