"""Y05-DirProbe fixed design and read-only table diagnostics (stdlib only)."""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, math, statistics

VERSION='Y05-DirProbe-1.0.1'
SEQS=('crowd2','crowd3','balloon2')
CARRIERS=('P_init','C_current')
ACTIONS=('skip','full','parallel','orthogonal','norm_parallel','norm_orthogonal','scrambled_parallel','scrambled_orthogonal')
HORIZONS=(4,8,16)
INSTALLED=('y05dirprobe_common.py','y05dirprobe_edit.py','gt4a2_y05dirprobe.py','test_y05dirprobe.py','y05dirprobe_identity.json')

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def jread(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def jwrite(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def cread(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cwrite(path,rows):
    require(bool(rows),'Empty table: '+str(path));p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(p)
def truth(x):return str(x).lower() in ('true','1','yes')
def pulse_due(index):return index%8==0 and index%32!=0

def anchors(oldplan,smoke=False):
    if smoke:
        return [dict(case_id=f'dirprobe_{s}_t{t:06d}',sequence=s,anchor=t,age=t,K=32,split='prefix_smoke') for s in SEQS for t in (8,24)]
    grouped=defaultdict(list)
    for row in oldplan['probe_plan']:grouped[row['sequence'],row['anchor']].append(row)
    require(len(grouped)==27,'Reviewed Y05 must supply exactly 27 unique tail anchors')
    result=[]
    for (s,t),rows in sorted(grouped.items(),key=lambda item:(SEQS.index(item[0][0]),item[0][1])):
        require(len(rows)==2 and {r['origin'] for r in rows}=={'A','S8'},'Duplicate/missing source parent origins')
        require(all(r['K']==32 and r['age']==t%32 and r['age'] in (8,16,24) for r in rows),'Anchor identity changed')
        result.append(dict(case_id=f'dirprobe_{s}_t{t:06d}',sequence=s,anchor=t,age=t%32,K=32,split='development_tail'))
    require(all(sum(r['sequence']==s for r in result)==9 for s in SEQS),'Expected 9 anchors/video')
    return result

def schedule(anchor,carrier,action):
    require(carrier in CARRIERS and action in ACTIONS and pulse_due(anchor),'Invalid branch specification')
    reset=(anchor//32+1)*32
    result=[]
    for h in range(17):
        index=anchor+h
        active=(action!='skip') if h==0 else pulse_due(index)
        head=('C_current' if h==0 else carrier) if active else ''
        result.append(dict(h=h,index=index,repair_applied=active,active_head=head,after_reset=index>=reset))
    return result

def windows(case,carrier,by):
    require(set(by)==set(ACTIONS),'Missing action branch')
    require(all(set(rr)==set(range(17)) for rr in by.values()),'Missing branch RGB')
    control=by['skip'];result=[]
    for action,rr in by.items():
        for h in range(17):
            require(truth(rr[h]['matched'])==truth(control[h]['matched']),'GT pairing differs')
            if truth(rr[h]['matched']):require(int(rr[h]['valid_pixels'])==int(control[h]['valid_pixels']),'Prediction changed GT mask')
            if truth(rr[h]['after_reset']) and truth(rr[h]['matched']):
                require(max(abs(float(rr[h][m])-float(control[h][m])) for m in ('AbsRel','RMSE','delta1'))<=2e-5,'History survived native full reset')
        for horizon in HORIZONS:
            hs=[h for h in range(1,horizon+1) if truth(control[h]['matched'])]
            require(hs,'Empty shared future GT window')
            delta=[float(rr[h]['AbsRel'])-float(control[h]['AbsRel']) for h in hs]
            pre=[h for h in hs if not truth(rr[h]['after_reset'])]
            r=dict(case_id=case['case_id'],sequence=case['sequence'],anchor=case['anchor'],age=case['age'],carrier=carrier,action=action,horizon=horizon,n_GT=len(hs),mean_AbsRel=statistics.mean(float(rr[h]['AbsRel']) for h in hs),mean_gain=-statistics.mean(delta),cumulative_gain=-sum(delta),harmed_fraction=sum(x>1e-8 for x in delta)/len(delta),harm_gt_0p01=sum(x>.01 for x in delta),pre_reset_gain=sum(float(control[h]['AbsRel'])-float(rr[h]['AbsRel']) for h in pre),n_pre_reset_GT=len(pre),current_AbsRel=rr[0]['AbsRel'],current_excluded=True,future_policy='carrier_S8')
            result.append(r)
    return result

def budget_audit(y05):
    result=[]
    for arm in CARRIERS[1:]+('P_closed_loop',):
        for r in cread(Path(y05)/'training'/arm/'curve.csv'):
            cuda=float(r['cuda_sec']);wall=float(r['step_wall_sec'])
            if not (math.isfinite(cuda) and math.isfinite(wall) and cuda>0 and wall>0) or cuda>wall+.05:
                result.append(dict(arm=arm,step=int(r['step']),cuda_sec=cuda,step_wall_sec=wall,reason='CUDA interval inconsistent with enclosing host step'))
    return dict(status='UNVERIFIED' if result else 'NO_INTERVAL_ANOMALY_DETECTED',anomalies=result,equal_compute_certified=False,interpretation='Read-only audit; neither deletes events nor relabels checkpoints; new diagnostic makes no equal-compute claim')

def age_decomposition(rows):
    base={(r['sequence'],int(r['index'])):r for r in rows if r['config_id']=='C_current_K32_S8_g1' and truth(r['scored']) and truth(r['matched'])}
    grouped=defaultdict(list);counts=defaultdict(int)
    for r in rows:
        action=r['config_id'].removesuffix('_K32_S8_g1');key=r['sequence'],int(r['index'])
        if action not in ('F_future','FH_future_harm') or r['config_id']!=action+'_K32_S8_g1' or key not in base:continue
        tag='pulse' if truth(r['repair_applied']) else ('propagation' if int(r['age'])>=8 else 'before_first_pulse')
        grouped[action,r['sequence'],tag].append(float(r['AbsRel'])-float(base[key]['AbsRel']));counts[action,r['sequence']]+=1
    output=[]
    for action in ('F_future','FH_future_harm'):
        for seq in SEQS:
            for tag in ('pulse','propagation','before_first_pulse'):
                xs=grouped[action,seq,tag];require(xs,'Missing age decomposition group')
                output.append(dict(action=action,sequence=seq,frame_group=tag,n=len(xs),mean_delta_vs_C=statistics.mean(xs),contribution_to_video_mean=sum(xs)/counts[action,seq]))
    return output

def aggregate(window_rows):
    grouped=defaultdict(list)
    for r in window_rows:grouped[r['carrier'],r['action'],r['sequence'],int(r['horizon'])].append(r)
    sequence=[]
    for key,rr in sorted(grouped.items()):
        sequence.append(dict(zip(('carrier','action','sequence','horizon'),key),n_cases=len(rr),**{m:statistics.mean(float(r[m]) for r in rr) for m in ('mean_AbsRel','mean_gain','harmed_fraction')}))
    macro=[];comparisons=[]
    for carrier in CARRIERS:
        for h in HORIZONS:
            for action in ACTIONS:
                rr=[r for r in sequence if r['carrier']==carrier and r['action']==action and r['horizon']==h]
                require(len(rr)==3,'Three-video macro missing')
                macro.append(dict(carrier=carrier,action=action,horizon=h,macro_AbsRel=statistics.mean(r['mean_AbsRel'] for r in rr),macro_gain=statistics.mean(r['mean_gain'] for r in rr),macro_harmed_fraction=statistics.mean(r['harmed_fraction'] for r in rr)))
            for action,reference in [('parallel','norm_parallel'),('orthogonal','norm_orthogonal'),('parallel','scrambled_parallel'),('orthogonal','scrambled_orthogonal')]:
                for seq in SEQS:
                    rr=[r for r in window_rows if r['carrier']==carrier and r['sequence']==seq and int(r['horizon'])==h]
                    by={(r['case_id'],r['action']):r for r in rr};ids=sorted({r['case_id'] for r in rr})
                    diff=[float(by[c,action]['mean_AbsRel'])-float(by[c,reference]['mean_AbsRel']) for c in ids]
                    comparisons.append(dict(carrier=carrier,action=action,reference=reference,sequence=seq,horizon=h,n_cases=len(ids),mean_delta=statistics.mean(diff),better_cases=sum(x<-1e-8 for x in diff),worse_cases=sum(x>1e-8 for x in diff)))
    return sequence,macro,comparisons
