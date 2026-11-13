"""Small independent, CPU-only integrity and leakage checks."""
import csv,hashlib,math
from pathlib import Path
def require_files(paths):
 missing=[str(p) for p in paths if not Path(p).is_file()]
 if missing:raise FileNotFoundError(', '.join(missing))
def unique(rows,keys):
 seen=set()
 for r in rows:
  k=tuple(r.get(x) for x in keys)
  if any(x is None or x=='' for x in k) or k in seen:raise ValueError('missing or duplicate key '+repr(k))
  seen.add(k)
def check_sha(path,expected):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 if h.hexdigest()!=expected:raise ValueError('SHA256 mismatch')
def check_times(xs):
 vals=[float(x) for x in xs]
 if any(not math.isfinite(x) for x in vals) or any(a>=b for a,b in zip(vals,vals[1:])):raise ValueError('timestamp nonfinite or not strictly increasing')
def no_leak(features):
 forbidden=('future','gt','sequence_id','dataset_id','frame_id','oracle')
 for r in features:
  if r.get('role')=='predictor' and any(x in r['name'].lower() for x in forbidden):raise ValueError('forbidden feature '+r['name'])
def required_fields(rows,keys):
 for row in rows:
  if any(k not in row or row[k] in ('',None) for k in keys):raise ValueError('missing required result fields')
