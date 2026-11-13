"""Protocol, split, mask-independent recovery and device-lock helpers; no Torch import."""
from pathlib import Path
import csv,hashlib,json,os,random,shutil,subprocess

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
def cases_for(manifests,policy):
    cases=[]
    for s in policy['old_sequences']:
        rr=manifests[s]
        for i in range(1,policy['train_prefix_stop_exclusive']-1):
            if active(i,policy['train_K'],policy['train_period']) and matched(rr[i]) and matched(rr[i+1]):
                cases.append(dict(id=f'{s}_{i:06d}',sequence=s,anchor=i,parent=i-1,horizon=1))
        require(any(c['sequence']==s for c in cases),'No H1 training pairs for '+s)
    return cases
def order_for(cases,n,seed,sequences):
    rng=random.Random(seed);groups={s:[c['id'] for c in cases if c['sequence']==s] for s in sequences}
    require(all(groups.values()),'Missing sequence in round-robin order');out=[];perms={}
    for i in range(n):
        s=sequences[i%len(sequences)];turn=i//len(sequences);epoch,pos=divmod(turn,len(groups[s]))
        if (s,epoch) not in perms:
            v=groups[s].copy();rng.shuffle(v);perms[s,epoch]=v
        out.append(perms[s,epoch][pos])
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
