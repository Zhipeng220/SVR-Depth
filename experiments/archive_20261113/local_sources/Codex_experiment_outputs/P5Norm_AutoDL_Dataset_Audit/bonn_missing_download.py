#!/usr/bin/env python3
"""Download eight absent frozen V5 training sequences to a new isolated directory."""
import os,sys,json,hashlib,urllib.request,zipfile,time,shutil
from pathlib import Path
from PIL import Image
root=Path('/root/autodl-tmp/datasets/p5norm_v5_missing_20261110')
root.mkdir(exist_ok=False);(root/'archives').mkdir();(root/'sequences').mkdir()
names=['balloon_tracking','kidnapping_box','moving_nonobstructing_box2','moving_obstructing_box2','placing_nonobstructing_box2','placing_obstructing_box','removing_nonobstructing_box2','removing_obstructing_box']
report={'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'source':'https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/','items':[],'status':'RUNNING'}
def save(): (root/'download_audit.json').write_text(json.dumps(report,indent=2))
save()
try:
 for n in names:
  name='rgbd_bonn_'+n;url='https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/'+name+'.zip';p=root/'archives'/(name+'.zip');tmp=p.with_suffix('.zip.part');item={'name':name,'url':url,'status':'STARTED'};report['items'].append(item);save();print('START',name,flush=True)
  if shutil.disk_usage(root).free<8*1024**3: raise RuntimeError('less than 8 GiB disk free')
  h=hashlib.sha256();total=0
  with urllib.request.urlopen(url,timeout=90) as response, tmp.open('xb') as out:
   expected=response.headers.get('Content-Length');item['http_content_length']=int(expected) if expected else None
   while True:
    b=response.read(1024*1024)
    if not b:break
    out.write(b);h.update(b);total+=len(b)
  if expected and total!=int(expected):raise RuntimeError('content length mismatch')
  tmp.rename(p);item.update(archive_bytes=total,sha256=h.hexdigest(),sha256_provenance='locally computed; no official checksum located');save()
  with zipfile.ZipFile(p) as z:
   bad=z.testzip()
   if bad:raise RuntimeError('CRC error '+bad)
   dest=(root/'sequences').resolve()
   for entry in z.infolist():
    target=(dest/entry.filename).resolve()
    if os.path.commonpath([str(dest),str(target)])!=str(dest) or ((entry.external_attr >> 16)&0o170000)==0o120000:raise RuntimeError('unsafe zip member')
    if entry.filename.split('/')[0]!=name:raise RuntimeError('unexpected archive root '+entry.filename)
   z.extractall(dest)
  seq=root/'sequences'/name;counts={};errors=[]
  for sub in ['rgb','depth']:
   paths=sorted((seq/sub).glob('*.png'));counts[sub]=len(paths)
   for im in paths:
    try:
     with Image.open(im) as x:x.load()
    except Exception as e:errors.append(str(im)+': '+str(e))
  if not counts['rgb'] or not counts['depth'] or errors:raise RuntimeError('image verification failed: '+str((counts,errors[:3])))
  item.update(status='PASS_ARCHIVE_CRC_AND_ALL_PNG_DECODE',counts=counts,png_errors=errors,free_bytes=shutil.disk_usage(root).free);save();print('DONE',name,counts,flush=True)
 report['status']='COMPLETE'
except Exception as e:
 report['status']='STOPPED';report['error']=repr(e);print('STOP',repr(e),flush=True)
finally:
 report['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');save()
