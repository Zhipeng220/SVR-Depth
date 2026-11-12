"""Protocol, split, mask-independent recovery and device-lock helpers; no Torch import."""
from pathlib import Path
import csv,hashlib,json,math,os,random,shutil,subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    require(bool(rows),'Empty table '+str(p));p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    with q.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    q.replace(p)
def matched(r):return r.get('gt_match_status','').upper()=='MATCHED'
def active(i,k=32,period=8):return i%k!=0 and i%period==0
def phase(i):return int(i)%8
def micro_due(i):return phase(i)!=0
def coverage_plan(manifests,policy):
    cases=[];coverage=[]
    for s in policy['old_sequences']:
        rr=manifests[s];stop=policy['train_prefix_stop_exclusive']
        require(len(rr)>=stop,'Training prefix too short: '+s)
        for h in range(1,8):
            indices=[i for i in range(stop) if phase(i)==h];eligible=[i for i in indices if matched(rr[i])]
            coverage.append(dict(sequence=s,phase=h,RGB=len(indices),matched_GT=len(eligible),missing_GT=len(indices)-len(eligible),
                minimum=policy['minimum_GT_per_sequence_phase'],passed=len(eligible)>=policy['minimum_GT_per_sequence_phase']))
            cases.extend(dict(id=f'{s}_{i:06d}',sequence=s,index=i,phase=h,split='prefix_current_GT') for i in eligible)
    return cases,coverage
def order_for(cases,n,seed,sequences):
    # 21 sequence x phase strata; counts differ by at most one update.
    slots=[(s,h) for h in range(1,8) for s in sequences]
    groups={key:[c['id'] for c in cases if (c['sequence'],c['phase'])==key] for key in slots}
    require(all(groups.values()),'Missing sequence/phase stratum');out=[];perms={}
    for i in range(n):
        key=slots[i%len(slots)];turn=i//len(slots);epoch,pos=divmod(turn,len(groups[key]))
        if (key,epoch) not in perms:
            v=groups[key].copy();random.Random(f'{seed}:{key}:{epoch}').shuffle(v);perms[key,epoch]=v
        out.append(perms[key,epoch][pos])
    return out
def verify_hashes(inputs):
    for p,d in inputs.items():require(Path(p).is_file() and sha(p)==d,'Pinned input changed/missing: '+p)
def check_recovery(rows,order):
    require([r['step'] for r in rows]==list(range(1,len(rows)+1)),'Recovery has duplicate/missing updates')
    require(all(r['case_id']==order[i] for i,r in enumerate(rows)),'Recovery order differs')
def gpu_lock_path(visible=None,inventory=None):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0') if visible is None else visible
    require(visible and ',' not in visible,'Set CUDA_VISIBLE_DEVICES to one GPU')
    if inventory is None:
        binary=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
        inventory=subprocess.run([binary,'--query-gpu=index,uuid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=15).stdout
    entries=[tuple(x.strip() for x in l.split(',',1)) for l in inventory.splitlines() if ',' in l]
    found=[u for i,u in entries if visible in (i,u,u.removeprefix('GPU-'))]
    require(len(found)==1,'Cannot resolve selected GPU UUID');u=found[0].removeprefix('GPU-')
    return u,Path('/tmp')/('svrdepth_diagnostics_'+hashlib.sha256(u.encode()).hexdigest()[:16]+'.lock')
def locate_y05r(explicit,identity,root=ROOT):
    dirs=[Path(explicit).resolve()] if explicit!='auto' else (root/'outputs').glob('gt4a2_y05r_*/validation')
    good=[]
    for d in dirs:
        if (d/'protocol.json').is_file() and sha(d/'protocol.json')==identity['source_y05r_protocol_sha256']:
            s=read(d/'summary.json');require(s['signature']==sha(d/'protocol.json') and s['status']=='Y05R_COMPLETE_NOT_METHOD_PASS','Y05-R incomplete')
            require((d/'training/C_star/final.pt').is_file(),'Keep full server C* checkpoint, not only review ZIP');good.append(d)
    require(len(good)>0,'Reviewed Y05-R C_star checkpoint missing; set Y05R_DIR to validation directory')
    return max(good,key=lambda p:(p/'summary.json').stat().st_mtime_ns)
def aggregate_rows(rows):
    by={}
    for r in rows:
        if r['AbsRel']!='':by.setdefault(r['sequence'],[]).append(float(r['AbsRel']))
    require(bool(by),'No scored recordings');return sum(sum(v)/len(v) for v in by.values())/len(by)

def state_fields(state):
    r,c,f,rays=state;require(len(f)==4,'State must have exactly four feature levels')
    return dict(radius=r,confidence=c,**{f'F{i}':x for i,x in enumerate(f)},rays=rays)
def state_versions(state):return {k:(id(v),v._version) for k,v in state_fields(state).items()}
def tensor_digest(x):
    x=x.detach().cpu().contiguous();h=hashlib.sha256(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.numpy().tobytes());return h.hexdigest()
def state_digest(state):return {k:tensor_digest(v) for k,v in state_fields(state).items()}
def compare_state(expected,actual):require(expected==actual,'Persistent C* trajectory changed')

def source_audit(args,out,policy,identity):
    """Strict read-only stdlib audit and coverage before any Torch/CUDA import."""
    y05r=locate_y05r(args.y05r_dir,identity);prior=read(y05r/'protocol.json');cp=y05r/'training/C_star/final.pt'
    verify_hashes({str(ROOT/p):h for p,h in identity['original_source_sha256'].items()})
    require(sha(cp)==identity['Cstar_file_sha256'],'Reviewed C* checkpoint file bytes differ')
    manifests={};inputs={str(cp):sha(cp),str(y05r/'protocol.json'):sha(y05r/'protocol.json')}
    for s in policy['old_sequences']:
        path=Path(prior['paths'][s+'_pair_csv']);require(path.is_file() and sha(path)==identity['reviewed_manifest_sha256'][s],'Reviewed pairing CSV changed: '+s)
        rows=csvread(path);require([int(r['frame_index_zero_based']) for r in rows]==list(range(len(rows))),'RGB indices missing or reordered')
        for r in rows[:policy['train_prefix_stop_exclusive']]:
            require(Path(r['rgb_path']).is_file(),'Missing prefix RGB: '+r['rgb_path'])
            if matched(r):require(Path(r['depth_path']).is_file(),'Missing matched prefix GT: '+r['depth_path'])
        manifests[s]=rows;inputs[str(path)]=sha(path)
    cases,coverage=coverage_plan(manifests,policy);csvwrite(out/'data_coverage.csv',coverage)
    obj=dict(experiment=policy['experiment'],Cstar_checkpoint=str(cp),Cstar_checkpoint_sha256=sha(cp),
        source_y05r_signature=sha(y05r/'protocol.json'),input_sha256=inputs,coverage_passed=all(r['passed'] for r in coverage),
        GT_coverage_scope='matched GT entries in fixed original RGB0..255 manifests; valid GT pixel masks checked before optimization',
        valid_samples=len(cases),sequence_phase_cells=len(coverage))
    write(out/'source_audit.json',obj);return y05r,prior,cases,coverage,obj
