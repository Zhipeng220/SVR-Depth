#!/usr/bin/env python3
"""Verify whole package before writing. Original model code is never overwritten."""
from pathlib import Path
import hashlib,json,os,shutil,sys

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install():
    pkg=Path(__file__).resolve().parent;root=Path.cwd().resolve();series='d13'
    if not (root/'velodepth/models').is_dir():raise SystemExit('Run installer from the velodepth repository root')
    records={}
    for line in (pkg/'SHA256SUMS').read_text().splitlines():
        digest,rel=line.split(maxsplit=1);rel=rel.strip();p=Path(rel)
        if p.is_absolute() or '..' in p.parts or rel in records:raise SystemExit('Unsafe/duplicate manifest entry')
        records[rel]=digest
    for rel,digest in records.items():
        p=pkg/rel
        if not p.is_file() or sha(p)!=digest:raise SystemExit('Package checksum mismatch: '+rel)
    # Only this release's isolated namespace is installed; stale 1.0 files may coexist after extraction.
    namespace=pkg/'experiments/p1'/(series+'_suite')
    expected={p.relative_to(pkg).as_posix() for p in namespace.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if not expected.issubset(records):raise SystemExit('Unmanifested installed code')
    repairs=pkg/'prerequisites/persistent_feature_repair.py';target=root/'velodepth/models/persistent_feature_repair.py'
    if sha(repairs)!='6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c':raise SystemExit('Pinned repair prerequisite differs')
    operations=[(pkg/rel,root/rel) for rel in sorted(expected)]+[(pkg/f'run_{series}.sh',root/f'run_{series}.sh')]
    if not target.exists():operations.append((repairs,target))
    elif sha(target)!=sha(repairs):raise SystemExit('Existing repair-head implementation differs; no files modified')
    # Validate every destination first, preventing a partial install on conflict.
    for src,dst in operations:
        if dst.exists() and dst.read_bytes()!=src.read_bytes():raise SystemExit('Refusing to overwrite changed installed file: '+str(dst))
    for src,dst in operations:
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            tmp=dst.with_name(dst.name+'.install.tmp');shutil.copy2(src,tmp);os.replace(tmp,dst)
    print('D13 Gemini RTDP-Safe 2.0.0 installed; independent namespace; pinned repair module restored only if absent.')
if __name__=='__main__':install()
