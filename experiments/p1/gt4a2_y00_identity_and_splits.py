#!/usr/bin/env python3
"""Y00: audit experiment identity and lock declared scene-level splits.

Standard library only; no training, model loading, inference, or GT metrics.
Run from the VeloDepth repo root. A pass verifies supplied declarations and
file identities, not the truth of scene IDs or upstream training provenance.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, shutil, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'Y00-1.0'
MODEL_SHA = '615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e'
P_SHA = '77a86cdf523f776608b6b591d6c6552002b90398e85d00d3a3fe29afe14a0a24'
LEGACY = {
    'crowd2': ('f7bf9226edcedf6c05d63938e29f92bf9364878c9291dbd79172c973c296f350', 895, 768),
    'crowd3': ('9f4da8360b8f6b3039330dc7fe9dba6dd7219d9427e774b89006b09269fa17ad', 854, 256),
    'balloon2': ('afbd46ebed192aaf69f580822993fb7c756c5656de0595f68143993859918b06', 469, 256),
}
SOURCE = ['experiments/p1/gt4a2_exp1_train_eval.py',
          'experiments/p1/gt2_fixed_clip_repair_train.py',
          'velodepth/models/decoder_velo.py', 'velodepth/models/persistent_feature_repair.py',
          'velodepth/models/velodepth.py', 'velodepth/models/encoder_velo.py']
SOURCE_HASHES = ['c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f',
                 'bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192',
                 '35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08',
                 '6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c',
                 '739c5f27a749174987b9a36edaac32fb98070e87dbef0a0a52f7430d61149479',
                 '3c5aa2a3a6dce03b68373953037fe8a0313a2cab36d66e5e3b88a56ec4a7ba8a']

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def resolve(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()

def template():
    art = [
        {'id':'y02a_script','kind':'script','path':'experiments/p1/gt4a2_y02a_cross_method_oracle.py','expected_sha256':'575bc8bdf48027fc54b898975cc1737da17cf34d40bbdb8832424f32862dd521'},
        {'id':'y04a_script','kind':'script','path':'experiments/p1/gt4a2_y04a_native_pre_readout_eval.py','expected_sha256':'6fbe1a2464a7f9756f9ea68cf05f3476a89b005b8e6f889dd1a0093f231c3116'},
        {'id':'y07w_script','kind':'script','path':'experiments/p1/gt4a2_y07w_same_head_no_writeback.py','expected_sha256':'e3ef2dae3bc387ddb9f9e11828d266bb14aaab9be1b17b9390666a8ac38ad358'},
        {'id':'p_weights','kind':'weights','path':'outputs/gt4a2_y04_validation_fix3_20261025_225902/training/P_multi/gt4a2_c3_C_final_step1000.pt','expected_sha256':P_SHA},
        {'id':'o_weights','kind':'weights','path':'outputs/gt4a2_y04_validation_fix3_20261025_225902/training/O_multi/Y04_O_multi_final_step1000.pt'},
        {'id':'o_method_script','kind':'script','path':'experiments/p1/gt4a2_y04_fair_multiscene.py','expected_sha256':'2a24e5a67487d55177e3374f9c6be5e92795b508dfcdb299cefe46e37cc31d4f'},
        {'id':'overlap_evidence','kind':'evidence','path':'FILL_OVERLAP_EVIDENCE_PATH'},
    ] + [{'id':'source_'+str(i),'kind':'script','path':p,'expected_sha256':SOURCE_HASHES[i]} for i,p in enumerate(SOURCE)]
    return {
        'version':VERSION, 'snapshot':os.environ.get('VDEEP_SNAPSHOT','FILL_SNAPSHOT_PATH'),
        'scene_manifest':'FILL_SCENE_MANIFEST_CSV',
        'legacy_pairs':{s:f'outputs/gt4a2_exp1/gt4a2_exp1_{s}_pairs_used.csv' for s in LEGACY},
        'artifacts':art,
        'methods':{
            'A':{'artifacts':['source_4','source_5'],'contract':'FILL_A_STATE_CONTRACT'},
            'O':{'artifacts':['o_weights','o_method_script'],'contract':'FILL_O_STATE_AND_READOUT_CONTRACT'},
            'P':{'artifacts':['p_weights','y04a_script'],'contract':'FILL_P_STATE_AND_READOUT_CONTRACT'},
            'N':{'artifacts':['p_weights','y07w_script'],'contract':'FILL_N_STATE_AND_READOUT_CONTRACT'}},
        'timing_boundary':'FILL_TIMING_INCLUDED_EXCLUDED_WARMUP_DEVICE_AND_SYNCHRONIZATION',
        'selection_budget':'FILL_DEV_GAIN_CHECKPOINT_K_SELECTION_BUDGET_AND_FINAL_TEST_RULE',
        'seeds':'FILL_EXISTING_AND_PLANNED_SEEDS',
        'dataset_provenance':'FILL_SCENE_ID_MAPPING_DATASET_VERSION_AND_PROJECT_OVERLAP_CHECK',
        'base_training_overlap':{'status':'unknown','evidence_artifact':'overlap_evidence',
                                 'notes':'FILL_VERIFIED_NO_OVERLAP_OR_UNCERTAIN_WITH_EVIDENCE'},
        'allow_uncertain_base_overlap':False,
    }

def filled(x):
    return isinstance(x,str) and bool(x.strip()) and 'FILL_' not in x

def inspect(config):
    errors, warnings, files = [], [], []
    def err(message): errors.append(message)
    def register(label, value, kind, expected=None):
        if not filled(value): err(f'{label}: path not filled'); return None
        p = resolve(value)
        if not p.is_file(): err(f'{label}: file missing: {p}'); return None
        sha = digest(p)
        if expected and sha != expected: err(f'{label}: SHA256 mismatch: {p}')
        item = {'label':label,'kind':kind,'path':str(p),'sha256':sha,'bytes':p.stat().st_size}
        files.append(item); return item
    if config.get('version') != VERSION: err('config version mismatch')
    snapshot = config.get('snapshot','')
    if filled(snapshot):
        root = resolve(snapshot)
        register('base_weights',str(root/'model.safetensors'),'weights',MODEL_SHA)
        register('base_config',str(root/'config.json'),'config')
    else: err('snapshot not filled')
    artifacts = {}
    for a in config.get('artifacts',[]):
        label = a.get('id','')
        if not filled(label) or label in artifacts:
            err(f'invalid or duplicate artifact id: {label}'); continue
        artifacts[label] = a
        if a.get('kind') not in ('script','weights','evidence','config','other'): err(f'{label}: invalid kind')
        if a.get('expected_sha256') and (len(a['expected_sha256']) != 64 or any(c not in '0123456789abcdef' for c in a['expected_sha256'])): err(f'{label}: invalid expected hash')
        register('artifact:'+label,a.get('path',''),a.get('kind','other'),a.get('expected_sha256'))
    for required in ['y02a_script','y04a_script','y07w_script','p_weights','o_weights']+[f'source_{i}' for i in range(len(SOURCE))]:
        if required not in artifacts: err(f'missing required artifact: {required}')
    # Reviewed identity checks are mandatory even if the configuration is edited.
    if 'p_weights' in artifacts:
        p = artifacts['p_weights'].get('path','')
        if filled(p) and resolve(p).is_file() and digest(resolve(p)) != P_SHA: err('P checkpoint differs from reviewed Y04 P')
    for arm in ('A','O','P','N'):
        m = config.get('methods',{}).get(arm,{})
        if not filled(m.get('contract','')): err(f'{arm}: state/readout contract not filled')
        refs = m.get('artifacts',[])
        if not refs or any(x not in artifacts for x in refs): err(f'{arm}: invalid artifact references')
    for arm, refs in {'O':{'o_weights'},'P':{'p_weights'},'N':{'p_weights','y07w_script'}}.items():
        if not refs.issubset(set(config.get('methods',{}).get(arm,{}).get('artifacts',[]))): err(f'{arm}: missing mandatory artifact reference')
    for field in ('timing_boundary','selection_budget','seeds','dataset_provenance'):
        if not filled(config.get(field,'')): err(f'{field}: not filled')
    overlap = config.get('base_training_overlap',{})
    if overlap.get('status') not in ('verified_no_overlap','uncertain'): err('base_training_overlap.status must be verified_no_overlap or uncertain')
    if overlap.get('status') == 'uncertain':
        warnings.append('Base pretraining overlap remains uncertain; no independent-domain/pretraining-exclusion claim.')
        if config.get('allow_uncertain_base_overlap') is not True: err('uncertain overlap requires explicit allow_uncertain_base_overlap=true')
    if not filled(overlap.get('notes','')): err('base overlap evidence notes not filled')
    if overlap.get('evidence_artifact') not in artifacts or artifacts.get(overlap.get('evidence_artifact'),{}).get('kind') != 'evidence': err('base overlap requires a referenced evidence artifact')
    for seq,(expected,n,tail) in LEGACY.items():
        f = register('legacy_pairs:'+seq,config.get('legacy_pairs',{}).get(seq,''),'manifest',expected)
        if f:
            with open(f['path'],encoding='utf-8-sig',newline='') as fp: rows = list(csv.DictReader(fp))
            if len(rows) != n: err(f'{seq}: expected {n} RGB records, got {len(rows)}')
            f.update({'rows':len(rows),'tail_start_index':tail,'role':'legacy_development_only','not_scene_blind':True})
    scene = register('scene_manifest',config.get('scene_manifest',''),'manifest')
    split_records=[]; splits={'train':0,'dev':0,'test':0}; scene_roles={}; video_roles={}; manifest_roles={}; seen_videos=set()
    if scene:
        with open(scene['path'],encoding='utf-8-sig',newline='') as fp:
            reader=csv.DictReader(fp)
            required={'dataset','scene_id','video_id','split','pair_manifest','source','project_overlap','base_overlap','start_index','end_index'}
            if not reader.fieldnames or not required.issubset(reader.fieldnames): err('scene CSV missing columns: '+','.join(sorted(required-set(reader.fieldnames or []))))
            else:
                for line,row in enumerate(reader,2):
                    if any(not filled(row.get(k,'')) for k in required): err(f'scene CSV line {line}: missing fields'); continue
                    split=row['split'].strip().lower(); ds=row['dataset'].strip().casefold(); sc=row['scene_id'].strip().casefold(); vid=row['video_id'].strip().casefold()
                    if split not in splits: err(f'scene CSV line {line}: invalid split'); continue
                    if row['project_overlap'].strip().lower() != 'no': err(f'scene CSV line {line}: project overlap must be verified no')
                    if row['base_overlap'].strip().lower() not in ('no','uncertain'): err(f'scene CSV line {line}: base overlap must be no or uncertain')
                    if row['base_overlap'].strip().lower() == 'uncertain' and not config.get('allow_uncertain_base_overlap'): err(f'scene CSV line {line}: uncertain base overlap not explicitly allowed')
                    if row['base_overlap'].strip().lower() == 'uncertain' and overlap.get('status') != 'uncertain': err(f'scene CSV line {line}: row uncertainty conflicts with global status')
                    if ds == 'bonn' and (sc in LEGACY or vid in LEGACY): err(f'scene CSV line {line}: legacy sequence cannot be a new scene')
                    if (ds,vid) in seen_videos: err(f'scene CSV line {line}: duplicate dataset/video_id')
                    seen_videos.add((ds,vid))
                    try:
                        start,end=int(row['start_index']),int(row['end_index'])
                        if start<0 or end<start: raise ValueError()
                    except ValueError: err(f'scene CSV line {line}: invalid inclusive indices'); continue
                    for name,key,bank in [('scene',(ds,sc),scene_roles),('video',(ds,vid),video_roles)]:
                        old=bank.setdefault(key,split)
                        if old!=split: err(f'{name} appears across splits: {key}')
                    mf=register(f'scene_row:{line}',row['pair_manifest'],'manifest')
                    if mf:
                        if mf['sha256'] in {v[0] for v in LEGACY.values()}: err(f'scene CSV line {line}: legacy pair manifest is not a new scene')
                        old=manifest_roles.setdefault(mf['sha256'],split)
                        if old!=split: err(f'identical pair manifest appears across splits: line {line}')
                        with open(mf['path'],encoding='utf-8-sig',newline='') as fp2:
                            count=sum(1 for _ in csv.DictReader(fp2))
                        if end>=count: err(f'scene CSV line {line}: end_index {end} exceeds manifest rows {count}')
                        row=dict(row); row['pair_manifest_sha256']=mf['sha256']; row['manifest_rows']=count
                    splits[split]+=1; split_records.append(row)
    for name,n in splits.items():
        if n==0: err(f'no declared {name} video')
    for name in ('dev','test'):
        if 0<splits[name]<5: warnings.append(f'{name}: {splits[name]} videos; planned target is at least 5, not a hard threshold')
    warnings.append('Scene IDs, source provenance and GT/manifest correctness are human declarations; identity checks do not prove true independence.')
    warnings.append('Legacy crowd2/crowd3/balloon2 and their tails remain development-only.')
    return {'status':'BLOCKED' if errors else 'PREFLIGHT_PASS','errors':errors,'warnings':warnings,'files':files,'split_counts':splits,'scenes':split_records}

def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init-config',help='write starter JSON to a new path; never overwrites')
    p.add_argument('--config',help='completed Y00 JSON config')
    p.add_argument('--snapshot',help='override snapshot in config')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--archive-weights',action='store_true',help='include weights in review zip; can be large')
    p.add_argument('--out-dir',default='outputs/gt4a2_y00_validation')
    a=p.parse_args()
    if a.init_config:
        dest=Path(a.init_config)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x',encoding='utf-8') as f: json.dump(template(),f,indent=2,ensure_ascii=False); f.write('\n')
        print(f'[TEMPLATE] {dest}'); return 0
    if not a.config: p.error('--config is required unless --init-config is used')
    cp=resolve(a.config)
    cfg=json.loads(cp.read_text(encoding='utf-8-sig'))
    if a.snapshot: cfg['snapshot']=str(resolve(a.snapshot))
    out=resolve(a.out_dir)
    if out.exists() and any(out.iterdir()): raise RuntimeError(f'Output directory not empty; use a new --out-dir: {out}')
    out.mkdir(parents=True,exist_ok=True)
    report=inspect(cfg)
    report.update({'version':VERSION,'script_sha256':digest(Path(__file__).resolve()),'config_source':str(cp),'config_source_sha256':digest(cp),
                   'snapshot_override':a.snapshot or None,'python':sys.version,'platform':platform.platform(),
                   'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'training_performed':False,'inference_performed':False})
    write_json(out/'y00_preflight.json',report)
    for e in report['errors']: print('[BLOCKED]',e,flush=True)
    for w in report['warnings']: print('[WARNING]',w,flush=True)
    if report['errors']: print(f'[BLOCKED] see {out}/y00_preflight.json'); return 2
    if a.preflight_only: print(f'[PREFLIGHT_PASS] {out}/y00_preflight.json; no freeze created'); return 0
    # Verify identities once more before writing the sealed bundle.
    for item in report['files']:
        if digest(Path(item['path'])) != item['sha256']: raise RuntimeError('Input changed during audit: '+item['path'])
    write_json(out/'y00_effective_config.json',cfg)
    with (out/'y00_files.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['label','kind','path','sha256','bytes']; w=csv.DictWriter(f,cols,extrasaction='ignore'); w.writeheader(); w.writerows(report['files'])
    summary={'status':'Y00_DECLARATIONS_LOCKED','version':VERSION,'split_counts':report['split_counts'],
             'base_training_overlap':cfg['base_training_overlap'],'archive_weights':a.archive_weights,
             'preflight_sha256':digest(out/'y00_preflight.json'),'effective_config_sha256':digest(out/'y00_effective_config.json'),
             'limitations':report['warnings'],'note':'A lock records supplied evidence; it does not independently prove scene or pretraining separation.'}
    write_json(out/'y00_summary.json',summary)
    zpath=out/'y00_review.zip'
    with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for name in ('y00_preflight.json','y00_effective_config.json','y00_files.csv','y00_summary.json'): z.write(out/name,name)
        z.write(Path(__file__).resolve(),'script/gt4a2_y00_identity_and_splits.py')
        for i,item in enumerate(report['files']):
            if item['kind']=='weights' and not a.archive_weights: continue
            path=Path(item['path']); z.write(path,f'inputs/{i:04d}_{path.name}')
    with zipfile.ZipFile(zpath) as z:
        bad=z.testzip()
        if bad: raise RuntimeError('ZIP integrity failure: '+bad)
    for item in report['files']:
        if digest(Path(item['path'])) != item['sha256']: raise RuntimeError('Input changed while archiving: '+item['path'])
    print(f'[Y00_DECLARATIONS_LOCKED] {zpath}'); print('[SHA256]',digest(zpath)); return 0

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as exc: print('[ERROR]',type(exc).__name__,str(exc),file=sys.stderr); sys.exit(1)
