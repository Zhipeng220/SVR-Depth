#!/usr/bin/env python3
"""Development-only real-interface canary; no independent-test claims."""
import argparse,csv,os,sys,time
from pathlib import Path
from contract import digest,read_csv,validate_manifest,require,unit_id
from prepare import PINS,BASE_SHA
from quality import run,write_json

P_SHA='77a86cdf523f776608b6b591d6c6552002b90398e85d00d3a3fe29afe14a0a24'
O_SHA='558e1850d7b08939bef0db0eb4f7a48fdff8a2fb58417cd4c380ee3c6a6f6948'
MANIFEST_SHA='f7bf9226edcedf6c05d63938e29f92bf9364878c9291dbd79172c973c296f350'

def units():
    a=dict(id=unit_id('A','native',16,0.,1),family='A',seed='native',k=16,gain=0.,frequency=1,point_ids=['smoke_A'],experiment='M01_M02')
    o=dict(id=unit_id('O_multi','20261025',16,1.,1),family='O_multi',seed='20261025',k=16,gain=1.,frequency=1,point_ids=['smoke_O'],experiment='M01_M02')
    p=dict(id=unit_id('P','20261025',16,1.,1),family='P',seed='20261025',k=16,gain=1.,frequency=1,point_ids=['smoke_P'],experiment='M01_M02')
    arms=[dict(id=f'M04__{a}__s20261025',family='P',seed='20261025',k=16,gain=1.,frequency=1,arm=a,experiment='M04') for a in ('OFF','OUTPUT_ONLY','F_ONLY','P_NATIVE')]
    return [a,o,p,*arms]

def smoke(snapshot,out,with_timing=False):
    root=Path.cwd().resolve();out=Path(out).resolve();require(not out.exists(),'Smoke output directory must be new')
    inputs={};missing=[]
    def add(p,expected=None):
        p=Path(p).resolve();require(p.is_file(),'Missing smoke input: '+str(p));h=digest(p);require(expected is None or h==expected,'Reviewed smoke input changed: '+str(p));inputs[str(p)]=h;return str(p)
    for rel,h in PINS.items():add(root/rel,h)
    for name in ('contract.py','prepare.py','quality.py','timing.py','report.py','m04_branch.py','smoke.py'):add(Path(__file__).parent/name)
    snapshot=Path(snapshot).resolve();add(snapshot/'config.json');add(snapshot/'model.safetensors',BASE_SHA)
    p=add(root/'outputs/gt4a2_y04_validation_fix3_20261025_225902/training/P_multi/gt4a2_c3_C_final_step1000.pt',P_SHA)
    o=add(root/'outputs/gt4a2_y04_validation_fix3_20261025_225902/training/O_multi/Y04_O_multi_final_step1000.pt',O_SHA)
    source=Path(add(root/'outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv',MANIFEST_SHA));rows=read_csv(source)[:40]
    require(len(rows)==40,'Crowd2 development prefix has fewer than 40 RGB')
    for r in rows:
        add(r['rgb_path']);status=r['gt_match_status'].strip().upper()
        if status=='MATCHED':add(r['depth_path'])
        elif status=='MISSING_DEPTH_FILE':
            path=Path(r['depth_path']).resolve();require(not path.exists(),'Missing depth unexpectedly exists');missing.append(str(path))
    out.mkdir(parents=True);manifest=out/'crowd2_prefix_pairs.csv'
    with manifest.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    validate_manifest(manifest);add(manifest)
    plan=dict(status='FROZEN_QUALITY_ONLY',role='development_smoke',snapshot=str(snapshot),depth_min_m=.0001,depth_max_m=100.,native_replay_rtol=1e-5,native_replay_atol=1e-4,
        checkpoints={'P':[dict(seed='20261025',path=p)],'O_multi':[dict(seed='20261025',path=o)],'O_fast':[]},
        sequences=[dict(sequence='crowd2_prefix',manifest=str(manifest),RGB=40,physical_scene_group='development_unknown',physical_scene_verified=False)],units=units(),input_sha256=inputs,expected_absent_depth=missing,
        limitations=['Old Crowd2 development prefix, single historical seed, no O_fast.','Not an independent test, formal M01/M02/M04, or method decision.'])
    path=out/'smoke_plan.json';write_json(path,plan);print('[DEV SMOKE] Crowd2 RGB0..39 only; no formal claims',flush=True)
    code=run(path,out/'quality')
    if code:return code
    if with_timing:
        from timing import run as timing_run
        code=timing_run(path,out/'quality',out/'timing')
        if code:return code
    from report import report
    report(path,out/'quality',out/'report',out/'timing' if with_timing else None)
    print('[DEV SMOKE COMPLETE]',out,flush=True);return 0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT',''));ap.add_argument('--out-dir',required=True);ap.add_argument('--timing',action='store_true');a=ap.parse_args()
    require(bool(a.snapshot),'VDEEP_SNAPSHOT required');return smoke(a.snapshot,a.out_dir,a.timing)
if __name__=='__main__':raise SystemExit(main())
