"""Y05-H4 protocol and stdlib utilities. Torch is loaded by server preflight only."""
from pathlib import Path
from types import SimpleNamespace
import csv,hashlib,importlib.util,json,math,os,random,sys,zipfile

VERSION='Y05-H4-1.0';SEQS=('crowd2','crowd3','balloon2')
N_RGB={'crowd2':895,'crowd3':854,'balloon2':469};TAIL={'crowd2':768,'crowd3':256,'balloon2':256}
HELPER_SHA='6fbe1a2464a7f9756f9ea68cf05f3476a89b005b8e6f889dd1a0093f231c3116'
MODEL_SOURCE={'velodepth/models/velodepth.py':'739c5f27a749174987b9a36edaac32fb98070e87dbef0a0a52f7430d61149479','velodepth/models/encoder_velo.py':'3c5aa2a3a6dce03b68373953037fe8a0313a2cab36d66e5e3b88a56ec4a7ba8a'}
FILES=('y05_common.py','y05_engine.py','y05_train.py','y05_eval.py','gt4a2_y05_future_utility.py','test_y05.py')
TRAIN_ARMS=('C_current','F_future','FH_future_harm','P_closed_loop')

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def jread(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def jwrite(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def cread(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cwrite(path,rows):
    require(bool(rows),'Empty output table: '+str(path));p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');keys=list(dict.fromkeys(k for r in rows for k in r))
    with t.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    t.replace(p)
def truth(x):return str(x).lower() in ('true','1','yes')
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def matched(row):return str(row.get('gt_match_status','')).strip().upper()=='MATCHED'
def pulse_due(index,k,period=8):return index%k!=0 and index%period==0
def arm_active(kind,index,k):return index%k!=0 and (kind=='continuous' or (kind=='S8' and pulse_due(index,k)))

def frozen_settings(smoke=False):
    return dict(version=VERSION,horizon=4,train_K=32,train_ages=[8,16,24],prefix_stop_exclusive=256,
        parent_origins=['A','S8'],parent_head='frozen reviewed P_multi; off-policy parent bank',
        init='same reviewed P_multi warmstart for every arm; inherited Y04 exposure retained',
        fixed_updates=2 if smoke else 1000,max_P_updates=8 if smoke else 4000,
        seed=20261028,learning_rate=1e-4,weight_decay=1e-4,clip_grad_norm=1.,
        amp_initial_scale=1024.,amp_growth_interval=2000,
        lambda_harm=1.,harm_margin_AbsRel=0.,mu_anchor_delta=1e-4,
        loss='GT-only-mask frame AbsRel; equal available future frames; then equal windows/video exposure',
        normalization='anchor delta-F squared energy / detached anchor input energy',
        checkpoint_interval=1 if smoke else 25,activation_checkpoint=True,
        compute_match='first crossing of FH committed CUDA-event full-training-step interval, including preparation/DIS and optimization, <=one final-step overshoot; cache/checkpoint costs and physical repeats separate',
        no_tail_selection=True,no_hyperparameter_grid=True,train_arms=list(TRAIN_ARMS),
        ordinary_P='head at anchor and every future non-keyframe; same parent/sample/GT; fixed-update and CUDA-budget snapshots',
        timing_repeats=1 if smoke else 3,timing_RGB=32 if smoke else 256,timing_warmup=8 if smoke else 32,
        timing_scope='three-video prefix; synchronized wall for DIS+model+depth output D2H; excludes RGB I/O/preprocess and GT/metrics',
        deployment='primary K32/S8; K16/S8 transfer; continuous fixed-K pressure; no Auto admission',
        final_checkpoint='fixed last update only; no validation checkpoint/gain/scene selection')

def prefix_samples(manifests=None,smoke=False):
    result=[]
    for seq in SEQS:
        for origin in ('A','S8'):
            for cycle in ([0] if smoke else range(0,256,32)):
                for age in (8,16,24):
                    anchor=cycle+age;end=anchor+4
                    require(end<256 and end<(anchor//32+1)*32,'Training window crossed split/reset')
                    if manifests is not None:
                        rows=manifests[seq]
                        if not matched(rows[anchor]) or not any(matched(rows[i]) for i in range(anchor+1,end+1)):continue
                    result.append(dict(case_id=f'fit_{seq}_{origin}_t{anchor:06d}',sequence=seq,origin=origin,K=32,
                        anchor=anchor,age=age,snapshot_index=anchor-1,horizon=4,split='prefix'))
    if smoke:
        # Canary needs four GT positions; select by pairing status, never by error.
        selected=[]
        for seq in SEQS:
            for origin in ('A','S8'):
                candidates=[c for c in result if c['sequence']==seq and c['origin']==origin and (manifests is None or all(matched(manifests[seq][c['anchor']+h]) for h in range(5)))]
                require(bool(candidates),'Smoke has no complete-GT training canary: '+seq)
                selected.append(candidates[0])
        result=selected
    return result

def probe_cases(smoke=False):
    result=[]
    for seq in SEQS:
        starts=[0] if smoke else [c for c in range(((TAIL[seq]+31)//32)*32,N_RGB[seq],32) if c+24+16<N_RGB[seq]]
        starts=starts if smoke else [starts[i] for i in (0,(len(starts)-1)//2,len(starts)-1)]
        for origin in ('A','S8'):
            for c in starts:
                for age in ([8] if smoke else (8,16,24)):
                    t=c+age;result.append(dict(case_id=f'probe_{seq}_{origin}_t{t:06d}',sequence=seq,origin=origin,K=32,
                        anchor=t,age=age,snapshot_index=t-1,horizon=16,split='smoke' if smoke else 'tail'))
    return result

def sample_order(samples,count,seed):
    # Exact six-slot video/origin round robin; deterministic within-slot shuffled epochs.
    groups={(s,o):[c['case_id'] for c in samples if c['sequence']==s and c['origin']==o] for o in ('A','S8') for s in SEQS}
    require(all(groups.values()),'Every video/origin needs training windows')
    slots=list(groups);permutations={};result=[]
    for step in range(count):
        key=slots[step%6];turn=step//6;ids=groups[key];epoch=turn//len(ids)
        pk=(key,epoch)
        if pk not in permutations:
            ids=list(ids);random.Random(f'{seed}:{key}:{epoch}').shuffle(ids);permutations[pk]=ids
        result.append(permutations[pk][turn%len(ids)])
    return result

def compute_crossed(seconds,target):return seconds>=target
def checked_curve(curve,order):
    require([r['step'] for r in curve]==list(range(1,len(curve)+1)),'Recovery has duplicate/missing updates')
    require(all(r['case_id']==order[i] for i,r in enumerate(curve)),'Recovery sample order changed')
    require(all(math.isfinite(r['cuda_sec']) and r['cuda_sec']>0 for r in curve),'Recovery timing invalid')

def completed_dir(root,prefix,explicit,status):
    paths=[Path(explicit).resolve()] if explicit!='auto' else list((root/'outputs').glob(f'gt4a2_{prefix}_*/validation'))
    found=[]
    for p in paths:
        if not (p/'summary.json').is_file():continue
        s=jread(p/'summary.json')
        if s.get('status')==status and not s.get('partial_smoke'):
            require(sha(p/'protocol.json')==s['signature'],f'{prefix}: protocol signature differs')
            if prefix in ('y03','y10'):require(s.get('completed_cases')==108 and s.get('numerical_failure_branches')==0,f'{prefix}: diagnostic incomplete/failed')
            found.append(p)
    require(bool(found),f'Completed {prefix} validation missing; set {prefix.upper()}_DIR')
    return max(found,key=lambda p:(p/'summary.json').stat().st_mtime_ns).resolve()

def preflight(args):
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root));sys.path.insert(0,str(root/'experiments/p1'))
    hp=root/'experiments/p1/gt4a2_y04a_native_pre_readout_eval.py'
    require(hp.is_file() and sha(hp)==HELPER_SHA,'Reviewed Y04-A helper missing/changed')
    spec=importlib.util.spec_from_file_location('y05_original_helper',hp);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    oldargs=SimpleNamespace(snapshot=args.snapshot,y04_dir=args.y04_dir,p_checkpoint='',seqs=list(SEQS),smoke_only=False,**{s+'_pair_csv':getattr(args,s+'_pair_csv') for s in SEQS})
    base,manifests,_,rt,head=helper.preflight(oldargs)
    require(rt.torch.cuda.is_available(),'Real Y05 requires CUDA')
    for rel,h in MODEL_SOURCE.items():require(sha(root/rel)==h,'Reviewed model differs: '+rel)
    y09=completed_dir(root,'y08cal_y09pareto',args.y09_dir,'DEVELOPMENT_SUITE_COMPLETE')
    prior={'y03':completed_dir(root,'y03',args.y03_dir,'DIAGNOSIS_COMPLETE_NOT_METHOD_PASS'),'y10':completed_dir(root,'y10',args.y10_dir,'DIAGNOSIS_COMPLETE_NOT_METHOD_PASS')}
    old=jread(y09/'protocol.json');oldsum=jread(y09/'summary.json');env=jread(y09/'environment.json')
    require(oldsum.get('quality_RGB_records')==79848,'Y09 suite not complete')
    torch=rt.torch;torch.backends.cudnn.benchmark=env['cudnn_benchmark'];torch.backends.cuda.matmul.allow_tf32=env['matmul_allow_tf32'];torch.backends.cudnn.allow_tf32=env['cudnn_allow_tf32']
    ofast=y09/'training/O_fast/final.pt';require(ofast.is_file(),'Y09 O_fast final.pt missing; retain server weights')
    of=torch.load(ofast,map_location='cpu',weights_only=False)
    require(of['signature']==oldsum['signature'] and rt.exp1.state_sha256(of['state'])==oldsum['newly_trained_heads']['O_fast']['final_sha'],'O_fast identity/tensor SHA differs')
    # Import only the pinned old output adapter; never change its source.
    oldmod=root/'experiments/p1/y08y09_models.py';expected=[v for k,v in old['input_sha256'].items() if Path(k).name==oldmod.name]
    require(len(expected)==1 and sha(oldmod)==expected[0],'Reviewed output adapter source differs')
    common=root/'experiments/p1/y08y09_common.py';expected=[v for k,v in old['input_sha256'].items() if Path(k).name==common.name]
    require(len(expected)==1 and sha(common)==expected[0],'Reviewed output adapter helper differs')
    import y08y09_models as oldmodels
    sources=[Path(__file__).parent/n for n in FILES]+[root/'run_y05.sh',hp,oldmod,root/'experiments/p1/y08y09_common.py']
    inputs={str(p.resolve()):sha(p) for p in sources}
    for rel,h in base['source_sha256'].items():inputs[str((root/rel).resolve())]=h
    extra=[Path(args.snapshot)/'model.safetensors',Path(args.snapshot)/'config.json',Path(base['checkpoint']),ofast,
           Path(args.y04_dir)/'y04_run_identity.json',Path(args.y04_dir)/'y04_summary.json']
    for d in [y09,*prior.values()]:extra += [d/'protocol.json',d/'summary.json']
    extra += [y09/'environment.json',y09/'y08y09_frame_metrics.csv']
    for p in extra:inputs[str(p.resolve())]=sha(p)
    for s,records in manifests.items():
        require(len(records)==N_RGB[s],'RGB count changed: '+s)
        mp=Path(getattr(args,s+'_pair_csv')).resolve();inputs[str(mp)]=sha(mp)
        for r in records:
            for key in ['rgb_path']+(['depth_path'] if matched(r) else []):
                p=Path(r[key]).resolve()
                if str(p) not in inputs:inputs[str(p)]=sha(p)
    for name,d in prior.items():require(jread(d/'protocol.json')['checkpoint_sha256']==helper.P_SHA,'Prior P identity changed: '+name)
    props=torch.cuda.get_device_properties(0)
    environment=dict(gpu_name=props.name,uuid=str(getattr(props,'uuid','')),total_memory=props.total_memory,torch=str(torch.__version__),cuda=str(torch.version.cuda),python=sys.version.split()[0],numpy=rt.np.__version__,opencv=rt.cv2.__version__,threads=4,cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    samples=prefix_samples(manifests,args.smoke_only);settings=frozen_settings(args.smoke_only)
    order=sample_order(samples,settings['max_P_updates'],settings['seed'])
    plan=dict(experiment='Y05',version=VERSION,partial_smoke=args.smoke_only,settings=settings,
        paths=dict(snapshot=str(Path(args.snapshot).resolve()),y04_dir=str(Path(args.y04_dir).resolve()),y09_dir=str(y09),y03_dir=str(prior['y03']),y10_dir=str(prior['y10']),**{s+'_pair_csv':str(Path(getattr(args,s+'_pair_csv')).resolve()) for s in SEQS}),
        checkpoint_sha256=helper.P_SHA,base_weights_sha256=helper.MODEL_SHA,source_y09_signature=oldsum['signature'],
        sample_plan=samples,probe_plan=probe_cases(args.smoke_only),sample_order=order,sample_order_sha256=canonical(order),
        data_split='additional training restricted to RGB0..255; inherited reviewed P training remains inherited; original tails development only',
        parent_bank='continuous frozen A/S8 carriers; common off-policy parents for all training arms; not on-policy retraining',
        input_sha256=inputs,environment=environment,
        limits=['First formal stage H4 only; no automatic H8 or hyperparameter changes','No scene-specific tail gain/checkpoint selection','No independent test, Auto stability or Y11 PASS','CUDA backward kernels may be nondeterministic; checkpoint restores state/order, not a bitwise trajectory promise'])
    refs={(r['variant'],r['sequence'],int(r['index'])):r for r in cread(y09/'y08y09_frame_metrics.csv') if r['variant'] in ('A_K16_g0_f1/raw','A_K32_g0_f1/raw','P_K16_g1_f1/raw','P_K32_g1_f1/raw','O_fast_K32_g1_f1/raw')}
    return SimpleNamespace(root=root,args=args,rt=rt,base=base,head_state=head,ofast_state=of['state'],oldmodels=oldmodels,manifests=manifests,plan=plan,refs=refs)

def archive(out):
    out=Path(out);archive_path=out/'y05_review.zip';tmp=archive_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for input_path in sorted(out.rglob('*')):
            relative=input_path.relative_to(out)
            if input_path.is_file() and input_path.suffix in ('.json','.jsonl','.csv','.md') and not any(x in relative.parts for x in ('cache','depth','gt')) and input_path.name!='failure.json':z.write(input_path,str(relative))
        for n in FILES:z.write(Path(__file__).parent/n,'source/'+n)
        launcher=Path(__file__).resolve().parents[2]/'run_y05.sh'
        z.write(launcher,'source/run_y05.sh')
    with zipfile.ZipFile(tmp) as z:require(z.testzip() is None,'Review CRC failed')
    tmp.replace(archive_path);return archive_path
