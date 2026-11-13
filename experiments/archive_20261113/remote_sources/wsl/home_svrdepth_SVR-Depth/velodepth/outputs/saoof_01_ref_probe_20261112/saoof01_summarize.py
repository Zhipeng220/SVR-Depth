"""Aggregate completed V1.1 Bonn5 predictions and measured latency."""
import csv,json,statistics
from pathlib import Path
p=Path('outputs/saoof_01_ref_probe_20261112'); f=p/'formal'
seq=json.loads((f/'metrics_per_sequence.json').read_text()); assert len(seq)==5
rows=list(csv.DictReader((f/'metrics_per_frame.csv').open(encoding='utf-8-sig'))); assert len(rows)==550
names=[x['sequence'] for x in seq]
for name in names:
 assert sum(r['sequence']==name for r in rows)==110
 assert len(list((f/name).glob('frame_*.npz')))==110
lat=json.loads((p/'latency'/'latency.json').read_text())['results']+json.loads((p/'latency_other'/'latency.json').read_text())['results']
assert len(lat)==60
def cfg_key(label,mode,alpha): return (label,mode,alpha)
latmap={}
for x in lat:
 label='low' if x['size']==[224,294] else 'mid' if x['size']==[294,392] else None
 assert label and x['frame_count']==48 and x['repeats']==3 and x['warmup_frames']==50
 k=cfg_key(label,x['mode'],x['alpha']); latmap.setdefault(k,[]).append(x)
assert len(latmap)==12 and all(len(v)==5 and {x['sequence'].removesuffix('_val') for x in v}==set(names) for v in latmap.values())
out=[]
for label in ('low','mid'):
 for mode in range(3):
  for alpha in (.2,.5):
   key=f'{label}_mode{mode}_a{alpha}_'; ls=latmap[cfg_key(label,mode,alpha)]
   out.append({'resolution':label,'mode':mode,'alpha':alpha,'macro_absrel':statistics.mean(x[key+'absrel'] for x in seq),'macro_rmse':statistics.mean(x[key+'rmse'] for x in seq),'macro_delta1':statistics.mean(x[key+'delta1'] for x in seq),'mean_e2e_ms_equal_sequence':statistics.mean(x['end_to_end_mean_ms'] for x in ls),'p50_ms_mean_per_sequence':statistics.mean(x['end_to_end_p50_ms'] for x in ls),'p95_ms_mean_per_sequence':statistics.mean(x['end_to_end_p95_ms'] for x in ls),'fps_from_equal_sequence_mean_ms':1000/statistics.mean(x['end_to_end_mean_ms'] for x in ls),'peak_gpu_mb_max':max(x['peak_gpu_memory_mb'] for x in ls)})
best=min(x['macro_absrel'] for x in out); qualifying=[x for x in out if x['macro_absrel']<=best*1.01]; selected=min(qualifying,key=lambda x:x['mean_e2e_ms_equal_sequence'])
base=statistics.mean(x['native_absrel'] for x in seq)
gate={}
for label in ('low','mid'):
 for mode in range(3):
  key=f'{label}_mode{mode}'; cs=[x[key+'_C'] for x in seq]; cd=[x[key+'_Cdynamic'] for x in seq]
  gate[key]={'sequences_ge_30pct':sum(v>=.3 for v in cs),'C':dict(zip(names,cs)),'Cdynamic':dict(zip(names,cd))}
props=[x for x in rows if int(x['age'])>0]; fallbacks={}
for label in ('low','mid'):
 fallbacks[label]={'propagation_frames':len(props),'fallback_frames':sum(x[label+'_status']!='ok' for x in props),'mean_corr_fraction_native':statistics.mean(int(x[label+'_corr_pixels'])/(434*574) for x in props)}
winning=min(out,key=lambda x:x['macro_absrel']); fixed_key=f"{winning['resolution']}_mode{winning['mode']}_a{winning['alpha']}_absrel"
individual=[{'sequence':x['sequence'],'native_absrel':x['native_absrel'],'best_dev_fixed_absrel':x[fixed_key],'relative_improvement':(x['native_absrel']-x[fixed_key])/x['native_absrel']} for x in seq]
decision={'protocol':'V1.1','status':'PREDICTION_AND_TIMING_COMPLETED','frames':550,'sequences':5,'preprocess_parity':json.loads((p/'reference_input_parity.json').read_text()),'baseline':{'macro_absrel':base,'macro_rmse':statistics.mean(x['native_rmse'] for x in seq),'macro_delta1':statistics.mean(x['native_delta1'] for x in seq)},'reference_raw':{label:{'macro_absrel':statistics.mean(x[label+'_ref_absrel'] for x in seq),'macro_rmse':statistics.mean(x[label+'_ref_rmse'] for x in seq),'macro_delta1':statistics.mean(x[label+'_ref_delta1'] for x in seq)} for label in ('low','mid')},'fixed_configs':out,'minimum_macro_absrel':best,'one_percent_cutoff':best*1.01,'qualifying_configs':qualifying,'selected_for_resolution_rule':selected,'chosen_resolution':selected['resolution'],'gate1':{'at_least_one_qualifies':any(x['sequences_ge_30pct']>=3 for x in gate.values()),'complementarity':gate,'fallback':fallbacks,'interpretation':'Engineering and information-source criterion only; not Gate 2 or cross-domain validation.'},'best_fixed_dev_vs_native':{'config':winning,'absolute_absrel_improvement':base-best,'relative_improvement':(base-best)/base,'per_sequence':individual},'latency_note':'48 measured frames per sequence/config: indices 0-15 repeated 3 times; 50-frame warmup once per config; includes HDF5 read/decode, official preprocess, keyframe=1 image and propagation=2 images plus native backward flow, native and reference inference, extra bidirectional DIS, correspondence, fusion, output geometry, CUDA sync; excludes model load and prediction cache writes. Per-sequence timings are from two runs, not per-frame full 110-frame production latencies. GPU component is only native+reference event sum; extra CPU flow excludes official-loader backward DIS.','provenance_note':'After prediction, a helper was added to the reference preprocessing module; reference_input implementation was unchanged and same-source parity was rerun. Full dependency-file hash was not recorded during prediction.'}
(p/'SAOOF01_decision.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding='utf8')
with (p/'SAOOF01_config_summary.csv').open('w',newline='',encoding='utf-8-sig') as h:
 w=csv.DictWriter(h,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
with (p/'SAOOF01_sequence_summary.csv').open('w',newline='',encoding='utf-8-sig') as h:
 keys=list(dict.fromkeys(k for x in seq for k in x)); w=csv.DictWriter(h,fieldnames=keys); w.writeheader(); w.writerows(seq)
print('baseline',base,'best',best,'relative',(base-best)/base,'chosen',selected,'Gate1 pixel criterion',decision['gate1']['at_least_one_qualifies'])
