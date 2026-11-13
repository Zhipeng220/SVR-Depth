import hashlib, json, zipfile
from pathlib import Path
root=Path(__file__).resolve().parent.parent
work=root/'m05x_work'; docs=root/'docs/m05x'; docs.mkdir(parents=True,exist_ok=True)
raw=[]
for path in sorted(work.glob('remote_inventory_raw*.txt')):
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('{'): raw.append(json.loads(line))
remote={x['path']:x for x in raw}
archive=root/'SVR-Depth_实验证据快照_20261109.zip'; z=zipfile.ZipFile(archive)
identities={}; completions={}
for variant,fragment in [('V4','y04_v4_bonn13_indomain_seed20261025_20261108_235748'),('V5','y04_v5_preserve_seed20261025_20261109_112607')]:
    for name,target in [('run_identity.json',identities),('completion.json',completions)]:
        n=next(n for n in z.namelist() if fragment in n and n.endswith('/'+name))
        target[variant]=json.loads(z.read(n))
        rp='/home/svrdepth/SVR-Depth/velodepth/outputs/'+fragment+'/'+name
        assert hashlib.sha256(z.read(n)).hexdigest()==remote[rp]['sha256']
    rp='/home/svrdepth/SVR-Depth/velodepth/outputs/'+fragment+'/final.pt'
    assert completions[variant]['checkpoint_sha256']==remote[rp]['sha256']
base='/home/svrdepth/SVR-Depth/velodepth/'
checks={'V4 trainer':('experiments/p1/gt4a2_y04_v4_bonn_train_v0_2.py',identities['V4']['trainer_sha256']),
        'V5 trainer':('experiments/p1/y04_v5_trainer_v2.py',identities['V5']['trainer_sha256']),
        'V5 loss':('experiments/p1/y04_v5_train_cycle.py',identities['V5']['v5_loss_sha256']),
        'manifest':('outputs/y04_v4_manifest_candidate_20261108_190940/manifest_identity.json',identities['V4']['candidate_manifest_sha256']),
        'initial head':('outputs/gt4a2_c3/gt4a2_c3_initial_head.pt',identities['V4']['init_sha256'])}
for key,(path,expected) in checks.items(): assert remote[base+path]['sha256']==expected,key
for path,expected in identities['V4']['source_sha256'].items():
    if base+path in remote: assert remote[base+path]['sha256']==expected,path
for field in ['training_sequences','development_sequences','candidate_manifest_sha256','pair_sha256','slots','order_sha256','seed','updates','K','tbptt','init_sha256','base_sha256']:
    assert identities['V4'][field]==identities['V5'][field],field
for seq,expected in identities['V4']['pair_sha256'].items():
    path=base+'outputs/y04_v4_manifest_candidate_20261108_190940/'+seq+'_pairs_v4_candidate.csv'
    assert remote[path]['sha256']==expected,seq
files=[]
for p,r in sorted(remote.items()):
    variant='V5' if ('v5' in p.lower() or 'V5' in p) else 'V4' if 'v4' in p.lower() else 'shared'
    files.append({'experiment_id':'Y04-V4/V5','path':p,'exists':r['exists'],'sha256':r.get('sha256'),'bytes':r.get('bytes'),'seed':20261025,'model_variant':variant,'source_commit':None,'protocol':'Bonn13 training K32/TBPTT4; development evaluation K16','audit_status':'observed_sha256' if r['exists'] else 'not_found_at_candidate_path'})
for n in z.namelist():
    if n.startswith('01_V4_V5/') and n.endswith('/protocol.json'):
        rp=base+n.removeprefix('01_V4_V5/')
        if rp in remote: assert remote[rp]['sha256']==hashlib.sha256(z.read(n)).hexdigest(),n
        files.append({'experiment_id':'Y04-V4/V5-development','path':str(archive)+'!/'+n,'exists':True,'sha256':hashlib.sha256(z.read(n)).hexdigest(),'seed':20261025,'model_variant':'V5' if 'y04_v5' in n else 'V4','source_commit':None,'protocol':'K16 development; see archived protocol','audit_status':'archived_snapshot_only'})
manifest={'status':'PARTIAL_BLOCKED','server':'192.168.0.236:2222','server_root':base,'source_commit':None,'git_status':'not a git repository; commit and dirty diff unavailable','files':files,'identity_common_fields_match':True,'all_13_training_pair_hashes_match':True,'identities':identities,'completions':completions,'verified_checks':checks,'blockers':['Full exact Git provenance unavailable on original server and local workspace.','Evaluation input frames and pixelwise masks not independently rehashed in this audit.','Pixelwise GT mask and current D1 tensor invariants cannot be independently reverified from historical CSV alone.']}
(docs/'frozen_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('files',len(files),'identity_and_checkpoints_match PASS')
