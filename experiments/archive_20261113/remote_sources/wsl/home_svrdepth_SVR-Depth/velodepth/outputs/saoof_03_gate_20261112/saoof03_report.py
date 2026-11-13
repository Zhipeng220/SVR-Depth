"""Aggregate completed three-seed results against frozen Bonn5 baselines."""
import json,csv,statistics
from pathlib import Path
root=Path('outputs/saoof_03_gate_20261112/formal/training')
assert json.loads((root/'status.json').read_text())['state']=='COMPLETED'
baseline=json.loads(Path('outputs/saoof_01_ref_probe_20261112/formal/metrics_per_sequence.json').read_text())
base={r['sequence']:r for r in baseline}; seeds=[20261112,20261113,20261114]
results={(m,s):json.loads((root/f'{m}_seed{s}'/'result.json').read_text()) for m in ('dual','fair') for s in seeds}
metric=lambda r:statistics.mean(x['absrel'] for x in r)
native=metric([{'absrel':r['native_absrel']} for r in baseline]); fixed=metric([{'absrel':r['mid_mode2_a0.5_absrel']} for r in baseline])
rows=[]
for s in seeds:
 d=results['dual',s]; f=results['fair',s]; ds={x['sequence']:x for x in d['sequences']}; fs={x['sequence']:x for x in f['sequences']};
 improve=sum(ds[n]['absrel']<base[n]['native_absrel'] for n in base); severe=sum(ds[n]['absrel']>base[n]['native_absrel']*1.05 for n in base)
 rows.append({'seed':s,'native_macro_absrel':native,'fixed_macro_absrel':fixed,'fair_macro_absrel':f['macro_absrel'],'dual_macro_absrel':d['macro_absrel'],'relative_gain_vs_native':1-d['macro_absrel']/native,'relative_gain_vs_fair':1-d['macro_absrel']/f['macro_absrel'],'relative_gain_vs_fixed':1-d['macro_absrel']/fixed,'improved_sequences':improve,'severely_worse_sequences':severe,'dual_best_epoch':d['best_epoch'],'fair_best_epoch':f['best_epoch']})
with (root/'gate2_by_seed.csv').open('w',newline='',encoding='utf-8-sig') as h:
 w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
sequence_rows=[]
for s in seeds:
 ds={x['sequence']:x for x in results['dual',s]['sequences']}; fs={x['sequence']:x for x in results['fair',s]['sequences']}
 for n in sorted(base):
  sequence_rows.append({'seed':s,'sequence':n,'native':base[n]['native_absrel'],'fixed':base[n]['mid_mode2_a0.5_absrel'],'fair':fs[n]['absrel'],'dual':ds[n]['absrel']})
with (root/'gate2_by_sequence.csv').open('w',newline='',encoding='utf-8-sig') as h:
 w=csv.DictWriter(h,fieldnames=list(sequence_rows[0])); w.writeheader(); w.writerows(sequence_rows)
summary={'native_macro_absrel':native,'fixed_macro_absrel':fixed,'dual_mean':statistics.mean(x['dual_macro_absrel'] for x in rows),'dual_sd':statistics.stdev(x['dual_macro_absrel'] for x in rows),'fair_mean':statistics.mean(x['fair_macro_absrel'] for x in rows),'fair_sd':statistics.stdev(x['fair_macro_absrel'] for x in rows),'seeds':rows,'interpretation':'Bonn5 is used for early stopping and fixed-configuration selection; this is a development-set result, not independent or cross-domain validation.'}
(root/'gate2_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
