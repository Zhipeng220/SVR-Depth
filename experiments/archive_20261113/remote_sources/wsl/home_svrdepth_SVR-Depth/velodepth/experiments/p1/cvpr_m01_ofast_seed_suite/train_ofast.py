#!/usr/bin/env python3
"""Train O_fast alone via pinned Y09; no held-out evaluation."""
from pathlib import Path
from types import SimpleNamespace
import argparse, fcntl, hashlib, json, os, sys, time, traceback

PINS={
'experiments/p1/gt4a2_y08cal_y09pareto.py':'ab680c054b3a9c09f6c935c2e4fb12c9995880207145e5adc8ba865ee0cf765e',
'experiments/p1/y08y09_common.py':'17756ccfee812a2c973be6d7c4905a1d71e42a434f39c25632c0dad76f91ccf3',
'experiments/p1/y08y09_models.py':'3b5e24786ab61c20c5b780e01cef2006a1a7868fa88b071ea63588d696d0e5b8',
'experiments/p1/gt4a2_y11_fallback.py':'e4c5d46cbe48b56c1631090959803f9d8db4690e6426aa60f8ff68318e0503ce'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(v,msg):
 if not v:raise RuntimeError(msg)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True,choices=[20261025,20261101,20261102]);p.add_argument('--out-dir',required=True);p.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT',''));p.add_argument('--resume',action='store_true');p.add_argument('--smoke-only',action='store_true');p.add_argument('--y04a-dir',default='auto');args=p.parse_args()
 root=Path.cwd().resolve();out=Path(args.out_dir).resolve();need(out.is_relative_to(root/'outputs') and out!=root/'outputs','output must be below outputs');need(not(args.resume and args.smoke_only),'smoke resume unsupported');need(args.snapshot,'VDEEP_SNAPSHOT required')
 for rel,h in PINS.items():need((root/rel).is_file() and sha(root/rel)==h,'missing or changed pinned source: '+rel)
 need(not (out/'m01_ofast_status.json').exists(),'output completed; do not overwrite')
 if out.exists():need(out.is_dir() and ((out/'protocol.json').is_file() if args.resume else not any(out.iterdir())),'output must be new or a resumable run')
 else:need(not args.resume,'cannot resume without output identity')
 out.mkdir(parents=True,exist_ok=True);sys.path[:0]=[str(root),str(root/'experiments/p1')]
 import gt4a2_y08cal_y09pareto as y09
 from gt4a2_y11_fallback import gpu_lock_path
 a=SimpleNamespace(snapshot=str(Path(args.snapshot).resolve()),y04_dir=str(root/'outputs/gt4a2_y04_validation_fix3_20261025_225902'),y04a_dir=args.y04a_dir,smoke_only=args.smoke_only,resume=args.resume,stage='train',out_dir=str(out),**{s+'_pair_csv':str(root/'outputs/gt4a2_exp1'/f'gt4a2_exp1_{s}_pairs_used.csv') for s in y09.SEQS})
 started=time.monotonic(); phase='preflight'; stamp=f'{time.time_ns()}_{os.getpid()}'
 with (out/'attempts.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'event':'STARTED','attempt':stamp,'seed':args.seed,'resume':args.resume,'smoke':args.smoke_only})+'\n')
 try:
  # Y09 locks the data and historical fixed order; all new initializations use a distinct seed.
  proto,manifests,refs,rt,pstate,ostate,helper=y09.preflight(a)
  proto['m01_ofast_training']={'version':'M01-OFAST-SEED-1.0','seed':args.seed,'initialization_seed':args.seed,'order_policy':'Y09 fixed 20261025 order for all O_fast seeds','runner_sha256':sha(__file__),'smoke_only':args.smoke_only,'selection':'final step only','pinned_sources':PINS,'independent_test_access':False}
  a.protocol=proto;ident=out/'protocol.json'
  if ident.exists():need(args.resume and y09.json_read(ident)==proto,'resume identity differs')
  else:need(not args.resume,'resume protocol missing');y09.json_write(ident,proto)
  signature=sha(ident);print('[PREFLIGHT PASS] O_fast seed',args.seed,'historical fixed training order',flush=True)
  uuid,lockpath=gpu_lock_path();phase='gpu_lock'
  with lockpath.open('a+') as lock:
   print('[GPU QUEUE] waiting for shared GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
   need(rt.torch.cuda.is_available(),'CUDA unavailable');device_uuid=str(getattr(rt.torch.cuda.get_device_properties(0),'uuid','')).removeprefix('GPU-');need(device_uuid==uuid,'CUDA device differs from GPU lock')
   rt.torch.set_num_threads(4);y09.SEED=args.seed; y09.seed(rt)
   phase='training_cache';cache_start=time.monotonic();paths=y09.build_training_cache(a,rt,manifests,out,pstate,ostate,signature,refs);cache_sec=time.monotonic()-cache_start
   phase='training';train_start=time.monotonic();state,summary=y09.train_head(a,rt,'O_fast',paths,out,signature,pstate,ostate);train_sec=time.monotonic()-train_start
   phase='input_revalidation'
   for path,h in proto['input_sha256'].items():need(Path(path).is_file() and sha(path)==h,'input changed: '+path)
   for rel,h in PINS.items():need(sha(root/rel)==h,'source changed: '+rel)
   cp=out/'training/O_fast/final.pt';need(cp.is_file(),'final missing')
   status={'status':'SMOKE_PASS' if args.smoke_only else 'TRAINING_ONLY_COMPLETE','seed':args.seed,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'training_summary':str(out/'training/O_fast/summary.json'),'protocol_sha256':signature,'cache_seconds_this_attempt':cache_sec,'train_seconds_this_attempt':train_sec,'total_seconds_this_attempt':time.monotonic()-started,'smoke_only':args.smoke_only,'evaluation_performed':False,'training_performed':True,'historical_input_provenance':'current inputs pinned; earlier P training bytes require independent evidence','fixed_training_order_seed':20261025}
   y09.json_write(out/'m01_ofast_status.json',status)
   with (out/'attempts.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'event':'COMPLETE','attempt':stamp,'elapsed_sec':time.monotonic()-started})+'\n')
   print('[SMOKE PASS]' if args.smoke_only else '[COMPLETE]','O_fast seed',args.seed,'checkpoint',cp,flush=True)
 except BaseException as e:
  failure={'phase':phase,'error':str(e),'traceback':traceback.format_exc(),'elapsed_sec':time.monotonic()-started,'seed':args.seed};(out/('failure_'+stamp+'.json')).write_text(json.dumps(failure,indent=2,ensure_ascii=False),encoding='utf-8')
  with (out/'attempts.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'event':'FAILED','attempt':stamp,'phase':phase,'error':str(e),'elapsed_sec':time.monotonic()-started})+'\n')
  raise
if __name__=='__main__':main()
