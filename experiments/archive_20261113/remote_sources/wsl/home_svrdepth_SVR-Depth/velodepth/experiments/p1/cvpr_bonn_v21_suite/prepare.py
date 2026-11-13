#!/usr/bin/env python3
"""Freeze Bonn-only M01/M02/M04 quality protocol before inspecting held-out scores."""
import argparse,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
from contract import FAMILIES,REQUIRED_DEV_REFERENCES,digest,read_json,read_csv,validate_config,validate_manifest,verify_training_pairs,reject_known_development_paths,require

PINS={
    'experiments/p1/gt4a2_y04_fair_multiscene_fix2.py':'4972e676e8f4b7c867615546eac34dcebce96c6b3e8cc0b5ad51e91fce14d291',
    'experiments/p1/gt4a2_exp1_train_eval.py':'c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f',
    'experiments/p1/gt2_fixed_clip_repair_train.py':'bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192',
    'experiments/p1/gt4a2_c3_closed_loop.py':'76327208acac4543b2a87ccba8b7af435e4c5ce65267b42052e51b435cbad7f5',
    'experiments/p1/y08y09_models.py':'3b5e24786ab61c20c5b780e01cef2006a1a7868fa88b071ea63588d696d0e5b8',
    'experiments/p1/gt4a2_y08cal_y09pareto.py':'ab680c054b3a9c09f6c935c2e4fb12c9995880207145e5adc8ba865ee0cf765e',
    'experiments/p1/y08y09_common.py':'17756ccfee812a2c973be6d7c4905a1d71e42a434f39c25632c0dad76f91ccf3',
    'experiments/p1/y11_common.py':'426e461dc296ecae05cd1783dc5ae42689eaedd1cfdc81906eee75fd50e0222b',
    'experiments/p1/y11_data.py':'4afea4f740d880aed26b055734ffeed1d7efea409f1591f99cdedb2e4a1487bc',
    'experiments/p1/gt4a2_y11_fallback.py':'e4c5d46cbe48b56c1631090959803f9d8db4690e6426aa60f8ff68318e0503ce',
    'experiments/p1/gt4a2_y04a_native_pre_readout_eval.py':'6fbe1a2464a7f9756f9ea68cf05f3476a89b005b8e6f889dd1a0093f231c3116',
    'experiments/p1/y10_runtime.py':'4c3802822dd1eafb22ed1288b8f0494387f113822c305b9598054cbdaa545ec2',
    'experiments/p1/y10_core.py':'220702c9198a90586a4f58a0a9d1ea374131421f57b943861289eca5096b1b4f',
    'velodepth/models/velodepth.py':'739c5f27a749174987b9a36edaac32fb98070e87dbef0a0a52f7430d61149479',
    'velodepth/models/encoder_velo.py':'3c5aa2a3a6dce03b68373953037fe8a0313a2cab36d66e5e3b88a56ec4a7ba8a',
    'velodepth/models/decoder_velo.py':'35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08',
    'velodepth/models/persistent_feature_repair.py':'6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c'}
BASE_SHA='615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e'
PROTOCOL_DOCUMENT_SHA='769cc8e889a6e6ef5e6cb603ff4fb62da835ef275a69c66e0014cb13a40e3336'

def freeze(config_path,output):
    root=Path.cwd().resolve();config_path=Path(config_path).resolve();c=read_json(config_path);units=validate_config(c);inputs={};missing=[]
    def add(p,expected=None):
        p=Path(p);require(p.is_absolute() and p.is_file(),'Missing/non-absolute input: '+str(p));p=p.resolve()
        actual=digest(p);require(expected is None or actual==expected,'Input hash differs: '+str(p));inputs[str(p)]=actual;return str(p)
    for rel,h in PINS.items():add((root/rel).resolve(),h)
    for name in ('contract.py','prepare.py','quality.py','timing.py','report.py','m04_branch.py','smoke.py'):
        add((Path(__file__).parent/name).resolve())
    snap=Path(c['snapshot']);require(snap.is_absolute(),'Snapshot path must be absolute');snap=snap.resolve()
    add(snap/'model.safetensors',BASE_SHA);add(snap/'config.json');c['snapshot']=str(snap)
    for key in ('matched_training_evidence','selection_evidence','calibration_evidence','independence_evidence','m04_state_audit'):
        c[key]=add(c[key])
    for method in FAMILIES[1:]:
        for cp in c['checkpoints'][method]:
            require(cp.get('sha256') and cp.get('training_record'),'Checkpoint SHA and training record required')
            cp['path']=add(cp['path'],cp['sha256']);cp['training_record']=add(cp['training_record'])
        require(len({x['path'] for x in c['checkpoints'][method]})==3 and len({x['sha256'].lower() for x in c['checkpoints'][method]})==3,'Distinct seeds require distinct checkpoint paths and bytes: '+method)
    verify_training_pairs(c,read_json(c['matched_training_evidence']))
    reference_hashes=set()
    refs=c.get('known_development_manifests',[])
    if c['role']=='heldout':require(REQUIRED_DEV_REFERENCES.issubset({x.get('sequence') for x in refs}),'Heldout requires all 3 Y04 and 5 Y11 development reference manifests')
    for ref in refs:
        ref['manifest']=add(ref['manifest']);rr=read_csv(ref['manifest']);require(rr,'Empty development reference manifest')
        for row in rr:
            path=Path(row['rgb_path']);require(path.is_absolute() and path.is_file(),'Missing reference RGB: '+str(path));path=path.resolve()
            if str(path) not in inputs:add(path)
            reference_hashes.add(inputs[str(path)])
        print('[REFERENCE AUDIT]',ref['sequence'],'RGB',len(rr),flush=True)
    for s in c['sequences']:
        s['pairing_audit']=add(s['pairing_audit'])
        s['manifest']=add(s['manifest']);rr=validate_manifest(s['manifest']);s['RGB']=len(rr)
        if c['role']=='heldout':reject_known_development_paths(rr)
        s['matched_GT']=sum(r['gt_match_status'].strip().upper()=='MATCHED' for r in rr)
        for r in rr:
            p=Path(r['rgb_path']).resolve()
            if str(p) not in inputs:add(p)
            if c['role']=='heldout':require(inputs[str(p)] not in reference_hashes,'Heldout RGB byte-identical to a known development/training frame: '+str(p))
            if r['gt_match_status'].strip().upper()=='MATCHED':
                p=Path(r['depth_path']).resolve()
                if str(p) not in inputs:add(p)
            elif r['gt_match_status'].strip().upper()=='MISSING_DEPTH_FILE':
                require(r.get('depth_path') and Path(r['depth_path']).is_absolute(),'Missing depth must retain original absolute path')
                p=Path(r['depth_path']).resolve();require(not p.exists(),'Previously missing GT exists: '+str(p));missing.append(str(p))
        print('[FREEZE]',s['sequence'],'RGB',len(rr),'matched',s['matched_GT'],flush=True)
    c['units']=units;c['input_sha256']=inputs;c['expected_absent_depth']=sorted(set(missing))
    c['status']='FROZEN_QUALITY_ONLY';c['not_formal_completion']=True;c['source_plan_document_sha256']=PROTOCOL_DOCUMENT_SHA;c['source_config_path']=str(config_path);c['source_config_sha256']=digest(config_path)
    c['frozen_at_utc']=datetime.now(timezone.utc).isoformat();c['pre_freeze_access']='Candidate GT and RGB bytes hashed; no model predictions or method scores computed by prepare.'
    c['limitations']=['Bonn only; cross-source deferred by user.','Metadata, heldout status and calibration are operator declarations, not automatically proven.',
        'Training, optical-flow warp error, submodule timing, and formal method decisions are not implemented.']
    output=Path(output);require(not output.exists(),'Refusing to overwrite frozen plan');output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(c,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    print('[FROZEN]',output,'units/sequence=',len(units),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();freeze(a.config,a.out)
if __name__=='__main__':main()
