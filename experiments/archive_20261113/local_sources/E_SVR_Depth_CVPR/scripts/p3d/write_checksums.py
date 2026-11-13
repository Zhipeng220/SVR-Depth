import hashlib
from pathlib import Path
root=Path.cwd();dest=root/'reports/p3d_dataaudit_p4a_preparation/SHA256SUMS';paths=[]
for folder in ['docs/p3d','docs/data_audit','docs/p4a','scripts/p3d','scripts/data_audit','scripts/p4a','tests/p3d','reports/p3d_dataaudit_p4a_preparation']:
 paths.extend(p for p in (root/folder).rglob('*') if p.is_file() and p!=dest and '__pycache__' not in p.parts)
lines=[]
for p in sorted(paths):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 lines.append(h.hexdigest()+'  '+p.relative_to(root).as_posix())
dest.write_text('\n'.join(lines)+'\n',encoding='utf8');print('CHECKSUMS',len(lines))
