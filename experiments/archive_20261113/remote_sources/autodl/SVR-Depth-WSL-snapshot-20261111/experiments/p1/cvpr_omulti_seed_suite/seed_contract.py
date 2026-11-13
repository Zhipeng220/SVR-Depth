"""Y04 matched-seed invariants; standard-library only."""
import hashlib,json,random
from pathlib import Path

SEEDS=(20261025,20261101,20261102)
ORDER_SHA={20261025:'aa7aff6213fdbf9ddce8703e2eb707e8cb8eab495a575f96698d9d25d9b9d934',
    20261101:'afed28e25737aa6355e0aad93a621dd24dae1b7b4718219ee0fae0efc7d8f47f',
    20261102:'d94beec434fc78a986875bc99d373585b241a4b962d23dd20002abc7bc4cf684'}
BASE_SHA='615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e'
HISTORICAL_P_SHA='77a86cdf523f776608b6b591d6c6552002b90398e85d00d3a3fe29afe14a0a24'
SLOTS=[(s,i) for s in ('crowd2','crowd3','balloon2') for i in range(8)]

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def order(seed):
    rng=random.Random(seed);r=[]
    while len(r)<1000:
        b=list(range(24));rng.shuffle(b);r.extend(b)
    return r[:1000]
def order_sha(seed):return hashlib.sha256(','.join(map(str,order(seed))).encode()).hexdigest()
def check_p_metadata(obj,seed,p_sha,pair_shas,initial_sha):
    require(seed in SEEDS,'Unknown seed')
    expected_seed=obj.get('y04_outer_seed')
    if expected_seed is None:require(seed==20261025 and p_sha==HISTORICAL_P_SHA,'P outer seed missing or unverified')
    else:require(int(expected_seed)==seed,'P outer seed differs')
    require(obj.get('mode')=='pre_readout' and obj.get('y04_label')=='Y04_P_multi' and obj.get('y04_kind')=='persistent','P checkpoint is not Y04 P_multi')
    require(int(obj.get('step_count',-1))==1000 and int(obj.get('optimizer_steps',-1))==1000,'P checkpoint not final 1000-step')
    require(obj.get('pretrained_weights_sha256')==BASE_SHA,'P base model differs')
    require(obj.get('y04_initial_head_sha256')==initial_sha,'P initial head differs')
    require(obj.get('y04_sample_order_sha256')==order_sha(seed)==ORDER_SHA[seed],'P outer sample order differs')
    require(obj.get('training_pair_csv_sha256_by_sequence')==pair_shas,'P training manifest hashes differ')
    require(obj.get('training_cycle_slots')==[list(x) for x in SLOTS],'P training slots differ')
    require(set(obj.get('training_sequences',[]))=={'crowd2','crowd3','balloon2'},'P training sequences differ')
    require(isinstance(obj.get('repair_head_state_dict'),dict) and obj['repair_head_state_dict'],'P repair state missing')

def supervision_manifest(manifests):
    rows=[]
    for seq,cycle in SLOTS:
        rr=manifests[seq]
        for i in range(cycle*32+1,cycle*32+32):
            row=rr[i]
            if str(row.get('gt_match_status','')).strip().upper()=='MATCHED':
                p=Path(row['depth_path']).resolve();require(p.is_file(),'Training GT missing: '+str(p))
                rows.append(dict(sequence=seq,index=i,depth_path=str(p),depth_sha256=sha(p)))
                if len(rows)%100==0:print('[SUPERVISION HASH]',len(rows),flush=True)
    data=json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    return rows,hashlib.sha256(data).hexdigest()

def training_inputs(manifests):
    paths=set()
    for seq in ('crowd2','crowd3','balloon2'):
        require(len(manifests[seq])>=256,'Training prefix shorter than 256 RGB')
        for row in manifests[seq][:256]:
            rgb=Path(row['rgb_path']).resolve();require(rgb.is_file(),'Training RGB missing: '+str(rgb));paths.add(rgb)
            if str(row.get('gt_match_status','')).strip().upper()=='MATCHED':
                gt=Path(row['depth_path']).resolve();require(gt.is_file(),'Training GT missing: '+str(gt));paths.add(gt)
    hashes={}
    for i,p in enumerate(sorted(paths),1):
        hashes[str(p)]=sha(p)
        if i%100==0 or i==len(paths):print('[INPUT HASH]',i,'/',len(paths),flush=True)
    return hashes

def incomplete_pair(seed,p_sha,o_sha,initial_sha,supervision_sha,exposure):
    return dict(seed=str(seed),P_checkpoint_sha256=p_sha,O_multi_checkpoint_sha256=o_sha,P_initial_head_sha256=initial_sha,O_multi_initial_head_sha256=initial_sha,
        P_sample_order_sha256=order_sha(seed),O_multi_sample_order_sha256=order_sha(seed),P_supervision_sha256='',O_multi_supervision_sha256=supervision_sha,
        P_steps=1000,O_multi_steps=1000,P_gt_exposures=exposure,O_multi_gt_exposures=exposure,checkpoint_rule='fixed_final_step',
        evidence_note='INCOMPLETE: P checkpoint pins pair manifests and exposure counts but does not independently prove historical GT bytes. P_supervision_sha256 must be filled only after a separate provenance/immutability audit; do not copy the O hash merely to pass the gate.')
