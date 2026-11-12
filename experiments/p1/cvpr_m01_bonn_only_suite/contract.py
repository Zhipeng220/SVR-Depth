"""Bonn-only v2.1 quality protocol: deterministic units and pre-test safeguards."""
import csv,hashlib,json,math,re
from pathlib import Path

FAMILIES=('A','O_fast','O_multi','P')
ARMS=('OFF','OUTPUT_ONLY','F_ONLY','P_NATIVE')
KNOWN_DEV={'crowd2','crowd3','balloon2','moving_nonobstructing_box','moving_obstructing_box','placing_nonobstructing_box','removing_nonobstructing_box','person_tracking2','person_tracking','synchronous'}
REQUIRED_DEV_REFERENCES={'crowd2','crowd3','balloon2','moving_nonobstructing_box','moving_obstructing_box','placing_nonobstructing_box','removing_nonobstructing_box','person_tracking2'}

def require(ok,msg):
    if not ok:raise ValueError(msg)

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_json(p):
    with Path(p).open(encoding='utf-8-sig') as f:return json.load(f)

def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def unit_id(family,seed,k,gain,frequency):
    return f'{family}__s{seed}__k{k}__g{gain:g}__f{frequency}'

def verify_counts(unit,n,counts):
    k=unit['k'];f=unit['frequency'];keys=(n+k-1)//k;fast=n-keys
    if unit['experiment']=='M04':
        arm=unit['arm'];expected_fast=fast*(2 if arm in ('OUTPUT_ONLY','F_ONLY') else 1)
        require(counts.get('physical_fast_steps')==expected_fast and counts.get('repair_head_calls')==(0 if arm=='OFF' else fast),'M04 actual call ledger differs')
        return
    due=sum(1 for i in range(n) if i%k>0 and (i%k-1)%f==0);m=unit['family']
    expected=dict(keyframe_calls=keys,flow_calls=fast,native_fast_calls=fast,repair_head_calls=due if m=='P' else 0,
        output_adapter_calls=due if m in ('O_multi','O_fast') else 0,additional_readout_passes=due if m=='O_multi' else 0,rejected_candidates=0)
    require(counts==expected,'M01/M02 actual call ledger differs')

def plan_units(c):
    seeds={m:[str(x["seed"]) for x in c["checkpoints"][m]] for m in FAMILIES[1:]}
    require(all(len(v)==3 and len(set(v))==3 for v in seeds.values()),"Each trainable family requires three distinct seeds")
    require(set(seeds["P"])==set(seeds["O_multi"]),"P and O_multi require corresponding seed identifiers")
    require(all(re.fullmatch(r"[A-Za-z0-9_-]+",s) for v in seeds.values() for s in v),"Unsafe seed identifier")
    points=c["points"];require(len(points)==4 and {p["family"] for p in points}==set(FAMILIES),"M01 requires exactly one point for each method")
    require(len({p["id"] for p in points})==4,"Point identifiers must be unique")
    units=[]
    for p in points:
        m=p["family"];gain=float(p["gain"]);freq=p["frequency"]
        require(p.get("k")==16 and p.get("alignment")=="raw" and freq==1 and p.get("semantics"),"M01 requires K16/f1/raw and documented semantics")
        require(gain==(0 if m=="A" else 1),"M01 gains differ from fixed protocol")
        require(p.get("role")==("primary" if m=="P" else "selected"),"M01 point role differs")
        for seed in (["native"] if m=="A" else seeds[m]):
            units.append(dict(id=unit_id(m,seed,16,gain,1),family=m,seed=seed,k=16,gain=gain,frequency=1,point_ids=[p["id"]],experiment="M01_M02"))
    return units

def validate_config(c):
    require(c.get("version")=="BONN-M01-ONLY-1","Wrong M01 configuration version")
    role=c.get("role");require(role in ("heldout","heldout_history_unverified"),"Explicit M01 role required")
    if role=="heldout":require(c.get("test_scores_unseen") is True and c.get("access_history_status")=="verified_unseen","Verified heldout requires confirmed unseen scores")
    else:require(c.get("test_scores_unseen") is False and c.get("access_history_status")=="unknown","Unverified history must remain explicitly unknown")
    require(c.get("calibration_audited") is True,"Bonn metric depth, registration, resize and mask audit required")
    require(c.get("data_provenance_audited") is True,"Known development overlap audit required")
    require(c.get("selection_frozen_before_test") is True,"M01 points must be frozen before test")
    require(c.get("depth_encoding")=="Bonn_registered_uint16_z_div5000","Unsupported depth contract")
    low=c.get("depth_min_m");high=c.get("depth_max_m")
    require(type(low) in (int,float) and type(high) in (int,float) and math.isfinite(low) and math.isfinite(high) and 0<low<high,"Freeze a common valid depth range")
    require(all(c.get(k) for k in ("selection_evidence","calibration_evidence","independence_evidence")),"M01 training, point, calibration and independence evidence required")
    seqs=c["sequences"];require(bool(seqs) and len({s["sequence"] for s in seqs})==len(seqs),"Unique recording identifiers required")
    for s in seqs:
        require(all(s.get(x) for x in ("sequence","physical_scene_group","recording_session","sensor_model","camera_setup","manifest","audit_notes","pairing_audit","pretraining_overlap")),"Incomplete recording metadata")
        require(s["pretraining_overlap"] in ("unknown","documented_overlap","documented_no_overlap"),"Invalid pretraining status")
        require(type(s.get("physical_scene_verified")) is bool,"Physical scene verification status required")
        require(re.fullmatch(r"[A-Za-z0-9_-]+",s["sequence"]) is not None,"Unsafe sequence")
        require(s["sequence"].lower().removeprefix("rgbd_bonn_") not in KNOWN_DEV,"Known development recording cannot be used")
    return plan_units(c)

def validate_manifest(path):
    rr=read_csv(path);require(bool(rr),'Empty manifest')
    require([int(r['frame_index_zero_based']) for r in rr]==list(range(len(rr))),'RGB indices must start at zero and be contiguous')
    prev=None;rgb_seen=set();depth_seen=set()
    for r in rr:
        st=r['gt_match_status'].strip().upper();require(st in ('MATCHED','UNMATCHED','MISSING_DEPTH_FILE'),'Unknown GT matching status')
        rgb=Path(r['rgb_path']);require(rgb.is_absolute() and rgb.is_file(),'Missing/non-absolute RGB: '+str(rgb))
        require(str(rgb.resolve()) not in rgb_seen,'RGB path reused');rgb_seen.add(str(rgb.resolve()))
        ts=float(r['rgb_timestamp']);require(math.isfinite(ts) and (prev is None or ts>prev),'RGB timestamps must strictly increase');prev=ts
        if st=='MATCHED':
            gt=Path(r['depth_path']);require(gt.is_absolute() and gt.is_file(),'Missing/non-absolute GT: '+str(gt))
            require(str(gt.resolve()) not in depth_seen,'GT path reused');depth_seen.add(str(gt.resolve()))
            dts=float(r['depth_timestamp']);require(math.isfinite(dts) and abs(dts-ts)*1000<=20+1e-7,'GT association exceeds 20ms')
            require(str(r.get('gt_reused','')).strip().lower() in ('false','0'),'GT-reuse field must explicitly be false')
    require(any(r['gt_match_status'].strip().upper()=='MATCHED' for r in rr),'No matched GT')
    return rr

def reject_known_development_paths(records):
    for r in records:
        for component in Path(r['rgb_path']).parts:
            if component.lower().removeprefix('rgbd_bonn_') in KNOWN_DEV:
                raise ValueError('Known development RGB path cannot be held out: '+r['rgb_path'])

def verify_training_pairs(c,evidence):
    """Cross-check recorded matching; evidence itself still requires provenance audit."""
    pairs=evidence.get('pairs',[]);items={str(x.get('seed')):x for x in pairs}
    ps={str(x['seed']):x for x in c['checkpoints']['P']};os_={str(x['seed']):x for x in c['checkpoints']['O_multi']}
    require(len(items)==len(pairs) and set(items)==set(ps)==set(os_),'Matched-training evidence must cover each seed exactly once')
    for seed,r in items.items():
        require(str(r.get('P_checkpoint_sha256','')).lower()==ps[seed]['sha256'].lower() and str(r.get('O_multi_checkpoint_sha256','')).lower()==os_[seed]['sha256'].lower(),'Evidence checkpoint SHA differs')
        for label in ('initial_head_sha256','sample_order_sha256','supervision_sha256'):
            a=str(r.get('P_'+label,''));b=str(r.get('O_multi_'+label,''))
            require(bool(re.fullmatch(r'[0-9a-fA-F]{64}',a)) and a.lower()==b.lower(),'P/O_multi matching evidence differs: '+label)
        require(type(r.get('P_steps')) is int and r['P_steps']==1000 and r['P_steps']==r.get('O_multi_steps'),'P/O_multi must use the matched Y04 1000 updates')
        require(type(r.get('P_gt_exposures')) is int and r['P_gt_exposures']>0 and r['P_gt_exposures']==r.get('O_multi_gt_exposures'),'P/O_multi GT exposures differ')
        require(r.get('checkpoint_rule')=='fixed_final_step','Final checkpoint rule must be fixed')
