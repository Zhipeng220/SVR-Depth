"""Pre-experiment phase statistics from reviewed N13 baseline CSVs, not new test results."""
from .common import *
import statistics

def reference_phase_audit(out,policy):
    rows=[]
    for seq in [*policy['old_sequences'],*policy['development_sequences']]:
        c=csvread(HERE/'reference_metrics/C_star'/(seq+'.csv'));a={int(r['index']):r for r in csvread(HERE/'reference_metrics/A'/(seq+'.csv'))}
        valid=[r for r in c if r['AbsRel']!='']
        for label in [*range(8),'K32_reset','S8_pulse','1..7']:
            rr=[r for r in valid if (micro_due(int(r['index'])) if label=='1..7' else frame_type(int(r['index']))==label if isinstance(label,str) else phase(int(r['index']))==label)]
            if not rr:continue
            delta=[float(r['AbsRel'])-float(a[int(r['index'])]['AbsRel']) for r in rr]
            rows.append(dict(sequence=seq,group='old_tail' if seq in policy['old_sequences'] else 'Y11_development',phase=label,frames=len(rr),AbsRel=statistics.mean(float(r['AbsRel']) for r in rr),severe_vs_A_count=sum(d>.01 for d in delta),severe_vs_A_rate=sum(d>.01 for d in delta)/len(delta)))
    macros=[]
    for group in ['old_tail','Y11_development']:
        for h in [*range(8),'K32_reset','S8_pulse','1..7']:
            rr=[r for r in rows if r['group']==group and r['phase']==h]
            if rr:macros.append(dict(group=group,phase=h,recordings=len(rr),AbsRel=statistics.mean(r['AbsRel'] for r in rr),severe_vs_A_rate=statistics.mean(r['severe_vs_A_rate'] for r in rr)))
    csvwrite(out/'cstar_phase_reference.csv',rows);csvwrite(out/'cstar_phase_reference_macro.csv',macros)
    png=HERE/'reference_cstar_phase.png'
    if png.exists():shutil.copy2(png,out/'reference_cstar_phase.png')
    write(out/'reference_phase_provenance.json',dict(source='reviewed N13 CSV baselines; motivation/development data already seen, not N14-v2 results',reviewed_N13_protocol_sha256=read(HERE/'identity.json')['reviewed_N13_protocol_sha256']))
    return rows,macros
