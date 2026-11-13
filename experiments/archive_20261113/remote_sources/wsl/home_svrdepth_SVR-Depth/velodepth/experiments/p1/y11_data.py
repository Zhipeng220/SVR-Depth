"""Prepare a fixed set of new Bonn recordings, with no model/GT-quality selection."""
from y11_common import *
from bisect import bisect_left,bisect_right
import argparse,fcntl,os,re,shutil,stat,time,urllib.request,zipfile

HERE=Path(__file__).resolve().parent

def source_y09(root,explicit='auto'):
    expected=jread(HERE/'y11_identity.json')['y09_protocol_sha256']
    choices=[Path(explicit).resolve()] if explicit!='auto' else sorted((root/'outputs').glob('gt4a2_y08cal_y09pareto_*/validation'))
    found=[]
    for d in choices:
        if (d/'protocol.json').is_file() and (d/'summary.json').is_file():
            s=jread(d/'summary.json')
            if sha(d/'protocol.json')==expected and s.get('status')=='DEVELOPMENT_SUITE_COMPLETE' and s.get('signature')==expected:found.append(d)
    require(bool(found),'Reviewed completed Y09 with server checkpoints missing; set Y09_DIR to its validation directory')
    return max(found,key=lambda d:(d/'summary.json').stat().st_mtime_ns).resolve()

def timestamp_rows(path,base,allow_missing=False):
    rows=[];previous=None
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        parts=line.split(maxsplit=1);require(len(parts)==2,'Malformed timestamp line '+str(path))
        t=float(parts[0]);require(math.isfinite(t) and (previous is None or t>previous),'Timestamp order/duplicate changed')
        p=(base/parts[1]).resolve();require(p.is_relative_to(base.resolve()),'Missing/unsafe image path '+str(p))
        require(allow_missing or p.is_file(),'Missing/unsafe image path '+str(p))
        rows.append((t,p));previous=t
    require(bool(rows),'Empty timestamp list '+str(path));return rows

def associate(rgb,depth,max_ms=20.,unavailable_depth_paths=()):
    """TUM-style global nearest one-to-one matching; retain every RGB."""
    ts=[t for t,_ in depth];candidates=[];delta=max_ms/1000.
    for i,(t,_) in enumerate(rgb):
        for j in range(bisect_left(ts,t-delta),bisect_right(ts,t+delta)):
            distance=abs(t-ts[j])
            if distance*1000<=max_ms:candidates.append((distance,i,j))
    used_rgb=set();used_depth=set();pairs={}
    for distance,i,j in sorted(candidates):
        if i not in used_rgb and j not in used_depth:
            pairs[i]=j;used_rgb.add(i);used_depth.add(j)
    rows=[];unavailable=set(map(str,unavailable_depth_paths))
    for i,(t,p) in enumerate(rgb):
        j=pairs.get(i);assigned=j is not None;missing=assigned and str(depth[j][1]) in unavailable
        matched=assigned and not missing
        rows.append(dict(frame_index_zero_based=i,rgb_timestamp=t,rgb_path=str(p),
            depth_timestamp=depth[j][0] if assigned else '',depth_path=str(depth[j][1]) if assigned else '',
            abs_delta_ms=abs(t-depth[j][0])*1000 if assigned else '',gt_match_status='MISSING_DEPTH_FILE' if missing else 'MATCHED' if matched else 'UNMATCHED',gt_reused=False))
    return rows

def audit_indexed_files(folder,seq,data_root,policy,download=False):
    """Restore local omissions; distinguish source-proven absent GT from broken local data.

    Raw index files are never edited. Missing GT keeps its original timestamp
    association; it is never replaced with a different nearby supervision frame.
    """
    folder=Path(folder).resolve();data_root=Path(data_root).resolve()
    rgb=timestamp_rows(folder/'rgb.txt',folder,allow_missing=True)
    depth=timestamp_rows(folder/'depth.txt',folder,allow_missing=True)
    missing_rgb=[p for _,p in rgb if not p.is_file()];missing_depth=[p for _,p in depth if not p.is_file()]
    report=dict(sequence=seq,rgb_index_entries=len(rgb),depth_index_entries=len(depth),
        initially_missing_RGB=[str(p) for p in missing_rgb],initially_missing_depth=[str(p) for p in missing_depth],
        restored_from_ZIP=[],source_proven_missing_depth=[],source_zip_CRC_verified=False,
        raw_index_sha256={n:sha(folder/n) for n in ('rgb.txt','depth.txt')})
    if not missing_rgb and not missing_depth:return report
    archive=data_root/'y11_downloads'/f'rgbd_bonn_{seq}.zip'
    if not archive.exists():
        require(download,'Indexed files are missing but the original ZIP is unavailable; rerun preparation with --download. No GT/RGB silently omitted.')
        print('[AUDIT DOWNLOAD] obtaining original ZIP to verify missing-file provenance:',seq,flush=True)
        download_archive(policy['data_url_base']+archive.name,archive)
    print('[DATA AUDIT] checking source ZIP for',len(missing_rgb),'RGB and',len(missing_depth),'depth omissions:',seq,flush=True)
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None,'Source ZIP CRC failed; no missing-GT interpretation is allowed')
        infos={i.filename:i for i in z.infolist()}
        require(len(infos)==len(z.infolist()),'Duplicate ZIP member names; ambiguous source provenance')
        prefixes=[n[:-len('rgb.txt')] for n in infos if n.endswith('rgb.txt') and n[:-len('rgb.txt')]+'depth.txt' in infos]
        prefixes=[p for p in prefixes if z.read(p+'rgb.txt')==(folder/'rgb.txt').read_bytes() and z.read(p+'depth.txt')==(folder/'depth.txt').read_bytes()]
        require(len(prefixes)==1,'Local raw rgb.txt/depth.txt do not uniquely match source ZIP; refusing to reinterpret missing files')
        prefix=prefixes[0];unavailable=[]
        for kind,paths in (('RGB',missing_rgb),('depth',missing_depth)):
            for path in paths:
                # Use the literal index spelling, not an arbitrary resolved symlink target.
                raw_name=next(line.split(maxsplit=1)[1] for line in (folder/('rgb.txt' if kind=='RGB' else 'depth.txt')).read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#') and (folder/line.split(maxsplit=1)[1]).resolve()==path)
                relative=Path(raw_name)
                require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe indexed ZIP path')
                member=prefix+relative.as_posix();info=infos.get(member)
                if info is None:
                    require(kind=='depth','Source ZIP itself lacks indexed RGB; complete trajectory cannot be reconstructed: '+member)
                    unavailable.append(str(path));continue
                require(not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16),'Indexed source member is not a regular image')
                path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.y11_restore.tmp')
                try:
                    with z.open(info) as source,temp.open('wb') as target:shutil.copyfileobj(source,target)
                    require(temp.stat().st_size==info.file_size,'Source restoration size mismatch');os.replace(temp,path)
                finally:temp.unlink(missing_ok=True)
                report['restored_from_ZIP'].append(dict(kind=kind,path=str(path),member=member,bytes=info.file_size,CRC32=f'{info.CRC:08x}'))
        report.update(source_zip=str(archive.resolve()),source_zip_sha256=sha(archive),source_zip_CRC_verified=True,
                      source_index_prefix=prefix,source_proven_missing_depth=unavailable)
    require(all(p.is_file() for _,p in rgb),'RGB still missing after source restoration')
    require(all(p.is_file() or str(p) in report['source_proven_missing_depth'] for _,p in depth),'Unexplained local GT loss')
    print('[DATA AUDIT]',seq,'restored',len(report['restored_from_ZIP']),'files; source-ZIP absent GT',len(report['source_proven_missing_depth']),'; ALL RGB retained',flush=True)
    return report

def prior_exposure_audit(root,sequences,exclude):
    """Evidence from prior result records; cannot certify unseen physical rooms or human history."""
    targets=set(sequences);hits=[];count=0;recorded={}
    outputs=root/'outputs'
    if not outputs.exists():return dict(files_scanned=0,conflicts=[],prior_file_sha256={})
    def contains(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k in targets:return True
                if k in ('sequence','sequence_name','seq') and isinstance(v,str) and v.removeprefix('rgbd_bonn_') in targets:return True
                if contains(v):return True
        elif isinstance(obj,list):return any(contains(v) for v in obj)
        elif isinstance(obj,str):
            if obj.removeprefix('rgbd_bonn_') in targets:return True
            return any(re.search(r'rgbd_bonn_'+re.escape(s)+r'(?:/|\\|$)',obj) for s in targets)
        return False
    for p in sorted(outputs.rglob('*')):
        if not p.is_file() or p.suffix not in ('.json','.csv'):continue
        # This suite's repeated invocations retain the same frozen point/data.
        # Its own previous records must not be mistaken for prior method selection.
        if p.is_relative_to(exclude) or any(x.startswith('gt4a2_y11_fallback_') for x in p.relative_to(outputs).parts):continue
        count+=1;found=False
        if p.suffix=='.json':
            try:found=contains(jread(p))
            except (ValueError,OSError):continue
        else:
            try:
                with p.open(encoding='utf-8-sig',newline='') as f:
                    for row in csv.DictReader(f):
                        if any(str(row.get(k,'')).removeprefix('rgbd_bonn_') in targets for k in ('sequence','sequence_name','seq')):found=True;break
                        if any(contains(row.get(k,'')) for k in ('rgb_path','depth_path')):found=True;break
            except (ValueError,OSError,csv.Error):continue
        recorded[str(p.resolve())]=sha(p)
        if found:hits.append(str(p.resolve()))
    return dict(files_scanned=count,conflicts=hits,prior_file_sha256=recorded,
                scope='local retained result records only; not a guarantee about unrecorded manual choices or inherited pretraining')

def locate_sequence(root,seq):
    choices=[root/f'rgbd_bonn_{seq}']
    if root.name==f'rgbd_bonn_{seq}':choices.append(root)
    found=[p.resolve() for p in choices if (p/'rgb.txt').is_file() and (p/'depth.txt').is_file()]
    if not found:
        found=[p.parent.resolve() for p in root.glob(f'**/rgbd_bonn_{seq}/rgb.txt') if (p.parent/'depth.txt').is_file()]
    unique=sorted(set(found));require(len(unique)<=1,'Dataset folder ambiguous: '+seq)
    return unique[0] if unique else None

def download_archive(url,target):
    part=target.with_suffix('.zip.part');target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        with zipfile.ZipFile(target) as z:require(z.testzip() is None,'Existing dataset ZIP CRC failed; retain and inspect '+str(target))
        return
    start=part.stat().st_size if part.exists() else 0
    request=urllib.request.Request(url,headers={'User-Agent':'SVR-Depth-Y11/1.0','Range':f'bytes={start}-'} if start else {'User-Agent':'SVR-Depth-Y11/1.0'})
    with urllib.request.urlopen(request,timeout=60) as response:
        append=start>0 and response.status==206
        if append:require(response.headers.get('Content-Range','').startswith(f'bytes {start}-'),'Download resume range mismatch')
        with part.open('ab' if append else 'wb') as f:
            while True:
                block=response.read(8<<20)
                if not block:break
                f.write(block)
    with zipfile.ZipFile(part) as z:require(z.testzip() is None,'Downloaded ZIP CRC failed')
    part.replace(target)

def obtain(root,seq,policy,download):
    current=locate_sequence(root,seq)
    if current:return current
    require(download,'Missing rgbd_bonn_'+seq+'; run prepare with --download or put its official data under '+str(root))
    archive=root/'y11_downloads'/f'rgbd_bonn_{seq}.zip'
    print('[DOWNLOAD]',seq,flush=True);download_archive(policy['data_url_base']+archive.name,archive)
    stage=root/f'.y11_extract_{seq}_{os.getpid()}';require(not stage.exists(),'Extraction staging collision');stage.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as z:
            required=sum(i.file_size for i in z.infolist());require(shutil.disk_usage(root).free>required+(1<<30),'Insufficient data disk for extraction')
            for info in z.infolist():
                p=(stage/info.filename).resolve()
                require(p.is_relative_to(stage.resolve()) and not stat.S_ISLNK(info.external_attr>>16),'Unsafe archive member')
            z.extractall(stage)
        current=locate_sequence(stage,seq)
        if current is None and (stage/'rgb.txt').is_file() and (stage/'depth.txt').is_file():current=stage
        require(current is not None,'Official archive layout not recognized; staging retained only until this invocation ends')
        target=root/f'rgbd_bonn_{seq}';require(not target.exists(),'Existing partial folder will not be overwritten: '+str(target))
        if current==stage:stage.rename(target)
        else:shutil.move(str(current),str(target))
        return target.resolve()
    finally:
        if stage.exists():shutil.rmtree(stage)

def prepare(root,data_root,out,y09_dir='auto',download=False):
    root=Path(root).resolve();data_root=Path(data_root).resolve();out=Path(out).resolve();data_root.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    policy=jread(HERE/'y11_policy.json');seqs=policy['confirmation_sequences']
    require(not(set(seqs)&set(policy['known_development_sequences'])),'Known development data cannot become independent confirmation')
    require(not(set(seqs)&set(policy['sealed_future_sequences'])),'Confirmation cannot touch sealed future recordings')
    y09=source_y09(root,y09_dir);prior=prior_exposure_audit(root,seqs,out)
    require(not prior['conflicts'],'Confirmation recording already appears in retained prior outputs: '+str(prior['conflicts'][:5]))
    frozen=out/'dataset_protocol.json'
    if frozen.exists():
        old=jread(frozen);require(old['policy_sha256']==sha(HERE/'y11_policy.json') and old['data_root']==str(data_root),'Frozen dataset/policy differs; never change it after viewing results')
        for path,digest in old['input_sha256'].items():require(Path(path).is_file() and sha(path)==digest,'Frozen data changed: '+path)
        for path in old.get('expected_missing_depth',[]):require(not Path(path).exists(),'GT availability changed after freeze: '+path)
        print('[DATA REUSE]',frozen,flush=True);return frozen
    # Old image hashes are already pinned by reviewed Y09; no exposure is inferred from filenames alone.
    oldplan=jread(y09/'protocol.json');old_rgb={v for k,v in oldplan['input_sha256'].items() if '/rgb/' in k.replace('\\','/')}
    inputs={};manifests={};seen_rgb={};seen_gt=set();download_info={};within_recording_duplicate_RGB=0;expected_missing_depth=[]
    for seq in seqs:
        folder=obtain(data_root,seq,policy,download)
        audit=audit_indexed_files(folder,seq,data_root,policy,download)
        expected_missing_depth+=audit['source_proven_missing_depth']
        audit_path=out/'data_integrity'/f'{seq}.json';jwrite(audit_path,audit);inputs[str(audit_path)]=sha(audit_path)
        if audit.get('source_zip'):inputs[audit['source_zip']]=audit['source_zip_sha256']
        rgb=timestamp_rows(folder/'rgb.txt',folder);depth=timestamp_rows(folder/'depth.txt',folder,allow_missing=True)
        rows=associate(rgb,depth,policy['max_timestamp_delta_ms'],audit['source_proven_missing_depth']);require(len(rows)>32,'Confirmation video too short: '+seq)
        require(sum(r['gt_match_status']=='MATCHED' for r in rows)>=16,'Insufficient fixed-mask GT: '+seq)
        for r in rows:
            p=Path(r['rgb_path']);digest=sha(p)
            require(digest not in old_rgb,'RGB byte overlap with reviewed development data: '+str(p))
            require(digest not in seen_rgb or seen_rgb[digest]==seq,'Duplicate RGB bytes across different confirmation recordings: '+str(p))
            within_recording_duplicate_RGB+=digest in seen_rgb;seen_rgb[digest]=seq;inputs[str(p)]=digest
            if r['gt_match_status']=='MATCHED':
                q=Path(r['depth_path']);require(str(q) not in seen_gt,'GT reused across confirmation recordings');seen_gt.add(str(q));inputs[str(q)]=sha(q)
        for p in (folder/'rgb.txt',folder/'depth.txt'):inputs[str(p.resolve())]=sha(p)
        manifest=out/'manifests'/f'{seq}.csv';cwrite(manifest,rows);inputs[str(manifest)]=sha(manifest)
        manifests[seq]=dict(path=str(manifest),RGB=len(rows),matched_GT=sum(r['gt_match_status']=='MATCHED' for r in rows),
            missing_depth_associations=sum(r['gt_match_status']=='MISSING_DEPTH_FILE' for r in rows),data_integrity_audit=str(audit_path),folder=str(folder))
        archive=data_root/'y11_downloads'/f'rgbd_bonn_{seq}.zip'
        if archive.exists():download_info[seq]=dict(official_url=policy['data_url_base']+archive.name,archive_sha256=sha(archive))
        print('[PAIRED]',seq,manifests[seq]['RGB'],'RGB;',manifests[seq]['matched_GT'],'GT',flush=True)
    obj=dict(version=VERSION,policy_sha256=sha(HERE/'y11_policy.json'),created_unix=time.time(),data_root=str(data_root),manifests=manifests,
        matching='global sorted nearest one-to-one on original full indices within20ms; source-ZIP-proven absent GT marked unavailable without reassignment; every RGB retained; no GT reuse',depth_scale=policy['depth_scale'],
        prior_exposure=prior,RGB_byte_overlap_with_Y09=False,within_recording_duplicate_RGB=within_recording_duplicate_RGB,downloads=download_info,independence_scope=policy['independence_scope'],
        inherited_pretraining_independence='not established',expected_missing_depth=expected_missing_depth,sealed_future_sequences_accessed=False,final_test=False,input_sha256=inputs)
    jwrite(frozen,obj);print('[DATA FROZEN]',frozen,flush=True);return frozen

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.');p.add_argument('--data-root',required=True);p.add_argument('--out',default='outputs/y11_confirmation_data');p.add_argument('--y09-dir',default=os.environ.get('Y09_DIR','auto'));p.add_argument('--download',action='store_true');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'.prepare.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare(a.repo,a.data_root,a.out,a.y09_dir,a.download)

if __name__=='__main__':main()
