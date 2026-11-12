"""Recover reproducible derived caches; never rewrite a source signature or checkpoint."""
from .common import *
from datetime import datetime

def repair_cache(ctx,cases):
    root=ctx.out/'cache';moves={};valid={};expected={(c['sequence'],c['index']):c for c in cases}
    metadata=set(root.rglob('*.json'))|{p.with_suffix('.json') for p in root.rglob('*.pt') if p.stem.isdigit() and len(p.stem)==6}
    for n,p in enumerate(sorted(metadata),1):
        if p.parent.name not in ctx.policy['old_sequences']:continue
        require(not p.is_symlink(),'Cache metadata must not be a symlink');reason=None
        try:obj=read(p)
        except (OSError,ValueError,UnicodeError):obj=None;reason='missing/invalid JSON'
        if isinstance(obj,dict) and obj.get('signature') not in (None,ctx.signature):raise RuntimeError('Foreign cache signature; no cache was moved: '+str(p))
        if not isinstance(obj,dict) or obj.get('signature')!=ctx.signature:reason=reason or 'incomplete metadata'
        if p.name=='complete.json':
            wanted=sorted(i for s,i in expected if s==p.parent.name)
            if reason is None and obj.get('indices')!=wanted:reason='completion cursor differs'
            if reason and p.is_file():moves[p]=reason
            continue
        if not(p.stem.isdigit() and len(p.stem)==6):continue
        case=expected.get((p.parent.name,int(p.stem)));require(case is not None,'Unexpected cache sample; inspect before continuing: '+str(p))
        tensor=p.with_suffix('.pt');require(not tensor.is_symlink(),'Cache tensor must not be a symlink')
        if reason is None and (obj.get('path')!=str(tensor) or obj.get('case_id')!=case['id'] or obj.get('age32')!=case['age32'] or obj.get('phase')!=case['phase'] or obj.get('Cstar_checkpoint_sha256')!=ctx.plan['Cstar_file_sha256']):reason='incomplete cache provenance'
        if reason is None and (not tensor.is_file() or obj.get('file_sha256')!=sha(tensor)):reason='missing/tampered Tensor cache'
        if reason:
            for q in (p,tensor,p.parent/'complete.json'):
                if q.is_file():moves[q]=reason
        else:valid[case['id']]=obj
        if n%8==0:print('[CACHE VERIFY]',n,'metadata units;',len(valid),'valid',flush=True)
    for p in root.rglob('*.tmp'):
        if p.parent.name in ctx.policy['old_sequences']:moves[p]='uncommitted temporary file'
    backup=None
    if moves:
        backup=ctx.out.parent/('cache_quarantine_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        for p,reason in sorted(moves.items()):
            dest=backup/p.relative_to(ctx.out);dest.parent.mkdir(parents=True,exist_ok=True);os.replace(p,dest);sync_directory(dest.parent);sync_directory(p.parent)
            print('[CACHE RECOVER]',p.relative_to(ctx.out),reason,flush=True)
        write(backup/'repair_manifest.json',dict(signature=ctx.signature,run=str(ctx.out),moves={str(p):r for p,r in moves.items()},source_files_modified=False,checkpoints_modified=False))
    write(ctx.out/'cache_recovery.json',dict(signature=ctx.signature,valid_pairs=len(valid),quarantined_files=len(moves),backup=str(backup) if backup else None))
    ctx._verified_cache={}
    for case_id,m in valid.items():
        p=Path(m['path']);s=p.stat();ctx._verified_cache[str(p)]=(s.st_size,s.st_mtime_ns,m['file_sha256'])
    return valid
