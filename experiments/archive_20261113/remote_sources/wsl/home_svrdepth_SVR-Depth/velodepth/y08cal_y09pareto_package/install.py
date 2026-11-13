#!/usr/bin/env python3
"""Install only the new suite files; never overwrite old Y04/Y07/Y08-H sources."""
from pathlib import Path
import hashlib

source=Path(__file__).resolve().parent
root=Path.cwd()
if not (root/'velodepth/models/velodepth.py').is_file():
    raise SystemExit('Run this installer from the velodepth repository root.')
names=('gt4a2_y08cal_y09pareto.py','y08y09_common.py','y08y09_models.py','test_y08y09.py')
allowed={f'experiments/p1/{n}':n for n in names}
allowed['run_y08cal_y09pareto.sh']='run_y08cal_y09pareto.sh'
records=[]
for line in (source/'y08y09_SHA256SUMS').read_text().splitlines():
    digest,target=line.split(maxsplit=1);target=target.strip()
    if target not in allowed:raise SystemExit(f'Unexpected installation target: {target}')
    data=(source/allowed[target]).read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest:raise SystemExit(f'Package integrity failed: {target}')
    records.append((target,data))
if set(target for target,_ in records)!=set(allowed):raise SystemExit('Incomplete installation manifest.')
for target,data in records:
    p=root/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
(root/'experiments/p1/y08y09_SHA256SUMS').write_bytes((source/'y08y09_SHA256SUMS').read_bytes())
print('安装完成。运行：bash run_y08cal_y09pareto.sh',flush=True)
