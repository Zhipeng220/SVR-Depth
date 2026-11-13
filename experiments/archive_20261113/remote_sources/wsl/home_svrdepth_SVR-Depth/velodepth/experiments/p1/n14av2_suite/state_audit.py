"""Separate full-RGB replay state comparison; never performed inside timed inference."""
from .common import *

FIELDS=['radius','confidence','F0','F1','F2','F3','rays']
def audit_all(ctx,smoke=False):
    from .evaluation import Predictor
    results=[];seqs=[ctx.old_sequences[0]] if smoke else list(ctx.manifests)
    for seq in seqs:
        rr=ctx.manifests[seq][:40 if smoke else None];reference=None
        for method in ['C_star',*ctx.policy['arms']]:
            p=ctx.out/'state_audit'/method/(seq+'.csv');mp=p.with_suffix('.json')
            if mp.exists():
                m=read(mp);require(m['signature']==ctx.signature and m['sha256']==sha(p) and m['RGB']==len(rr) and m['passed'],'State audit unit changed')
                rows=csvread(p)
            else:
                pred=Predictor(ctx,method,seq,audit=True);rows=[]
                try:
                    for i,rec in enumerate(rr):
                        _,meta=pred.step(i,rec);actual=meta['state_hashes']
                        require(set(actual)==set(FIELDS),'State hash fields changed')
                        if reference is not None:compare_state({k:reference[i][k] for k in FIELDS},actual)
                        rows.append(dict(method=method,sequence=seq,index=i,phase=phase(i),state_equal=True,**actual))
                        if i==0 or (i+1)%200==0 or i==len(rr)-1:print('[STATE AUDIT]',method,seq,f'{i+1}/{len(rr)}',flush=True)
                    csvwrite(p,rows);m=dict(signature=ctx.signature,method=method,sequence=seq,RGB=len(rr),passed=True,sha256=sha(p));write(mp,m)
                finally:pred.close()
            require([int(r['index']) for r in rows]==list(range(len(rr))),'State audit RGB coverage incomplete')
            if method=='C_star':reference=rows
            else:
                for a,b in zip(reference,rows):compare_state({k:a[k] for k in FIELDS},{k:b[k] for k in FIELDS})
            results.append(m)
    result=dict(signature=ctx.signature,passed=True,records=results,state_fields=FIELDS,scope='independent replay per C*/A0/A1; all RGB including missing GT; before/after readout and against C*; no hashes inside latency measurement')
    write(ctx.out/'state_invariance.json',result);return result
