from pathlib import Path
import tarfile,sys
p=Path('/home/svrdepth/SVR-Depth/velodepth/outputs/p5norm_eng09_velo_raw_stagea/raw');archive=p/'rgbd_dataset_freiburg3_walking_halfsphere.tgz';dest=p/'extracted';dest.mkdir(exist_ok=False);base='rgbd_dataset_freiburg3_walking_halfsphere'
with tarfile.open(archive,'r:gz') as tar:
 members=tar.getmembers()
 for m in members:
  q=Path(m.name)
  if q.is_absolute() or '..' in q.parts or not q.parts or q.parts[0]!=base or not (m.isfile() or m.isdir()):raise RuntimeError('unsafe archive member '+m.name)
 tar.extractall(dest,members=members,filter='data')
print('EXTRACTED',len(members),'members')
