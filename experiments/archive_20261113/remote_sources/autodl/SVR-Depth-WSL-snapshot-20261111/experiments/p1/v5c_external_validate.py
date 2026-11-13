import collections,csv,json,math,statistics,sys
from pathlib import Path
name=sys.argv[1];p=Path(sys.argv[2]);root=Path.cwd();old=root/'outputs/y04_v4_external_compare_20261109_084229'/name
protocol=json.loads((p/'protocol.json').read_text());original=json.loads((old/'protocol.json').read_text())
for k in ('hdf5_sha256','baseline_model_sha256','checkpoint_sha256','depth_scale','depth_min_exclusive','depth_max_exclusive','crop','K','amp'):assert protocol[k]==original[k],k
arms={name:list(csv.DictReader((p/(name+'.csv')).open())) for name in ('v4_O','v5_O','v5_P')};legacy={name:list(csv.DictReader((old/(name+'.csv')).open())) for name in ('baseline','bonn13')}
oldmap={name:{(r['sequence'],r['index']):r for r in rows} for name,rows in legacy.items()}
count=len(arms['v5_P']);assert all(len(rows)==count for rows in arms.values());seqs=collections.defaultdict(list);statuses=collections.Counter();maxkey=0.
for a,b,c in zip(arms['v4_O'],arms['v5_O'],arms['v5_P']):
 key=(a['sequence'],a['index']);x=oldmap['baseline'][key];y=oldmap['bonn13'][key];allrows={'A':x,'v4_O':a,'v4_P':y,'v5_O':b,'v5_P':c}
 for r in allrows.values():assert all(r[k]==x[k] for k in ('sequence','index','keyframe','valid_pixels','mask_sha256'))
 statuses[tuple(r['status'] for r in allrows.values())]+=1
 if all(r['status']=='ok' for r in allrows.values()):
  seqs[a['sequence']].append(allrows)
  if a['keyframe']=='True':maxkey=max(maxkey,max(abs(float(x[k])-float(r[k])) for r in allrows.values() for k in ('AbsRel','RMSE','delta1')))
assert maxkey<2e-5
result={'dataset':name,'frames':count,'sequences_with_valid_pairs':len(seqs),'statuses':{str(k):v for k,v in statuses.items()},'max_keyframe_metric_diff':maxkey,'metrics':{},'per_sequence':{}}
for seq,items in seqs.items():
 result['per_sequence'][seq]={'matched':len(items),'arms':{arm:{k:statistics.mean(float(v[arm][k]) for v in items) for k in ('AbsRel','RMSE','delta1')} for arm in ('A','v4_O','v4_P','v5_O','v5_P')}}
for k in ('AbsRel','RMSE','delta1'):
 result['metrics'][k]={arm:statistics.mean(x['arms'][arm][k] for x in result['per_sequence'].values()) for arm in ('A','v4_O','v4_P','v5_O','v5_P')}
(p/'paired_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('PAIRED_PASS',name,'frames',count,'maxkey',maxkey,'metrics',result['metrics'],flush=True)
