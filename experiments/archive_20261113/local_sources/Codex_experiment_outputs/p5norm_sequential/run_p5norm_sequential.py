"""Fail-closed sequential orchestration scaffold; no model or CUDA imports."""
import argparse, json, pathlib, subprocess, sys, os, time, hashlib
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--worker',default=str(pathlib.Path(__file__).with_name('p5norm_strategy_worker.py')));a=p.parse_args()
 cfg=json.loads(pathlib.Path(a.config).read_text());out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
 if not cfg.get('anchors') or not isinstance(cfg['anchors'],list):raise ValueError('nonempty anchors required')
 manifest={'status':'STARTED','anchors':cfg['anchors'],'config_sha256':hashlib.sha256(pathlib.Path(a.config).read_bytes()).hexdigest(),'events':[]};(out/'run.json').write_text(json.dumps(manifest,indent=2))
 for idx,anchor in enumerate(cfg['anchors']):
  for strategy in 'ABC':
   directory=out/f'anchor_{idx:04d}'/('strategy_'+strategy);directory.mkdir(parents=True)
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';cmd=[sys.executable,a.worker,'--strategy',strategy,'--anchor-json',json.dumps(anchor),'--output',str(directory)]
   with (directory/'stdout.log').open('w') as log:
    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);ret=process.wait()
   manifest['events'].append({'strategy':strategy,'pid':process.pid,'returncode':ret,'anchor':anchor,'completed_at':time.time()});(out/'run.json').write_text(json.dumps(manifest,indent=2))
   if ret!=0:manifest['status']='BLOCKED_WORKER_FAILED';(out/'run.json').write_text(json.dumps(manifest,indent=2));return ret or 1
   result=directory/'manifest.json'
   if not result.is_file():manifest['status']='BLOCKED_MISSING_MANIFEST';(out/'run.json').write_text(json.dumps(manifest,indent=2));return 1
   record=json.loads(result.read_text())
   if record.get('strategy')!=strategy or record.get('anchor')!=anchor or record.get('status')!='PASS':manifest['status']='BLOCKED_INVALID_MANIFEST';(out/'run.json').write_text(json.dumps(manifest,indent=2));return 1
 manifest['status']='BLOCKED_EQUIVALENCE_NOT_IMPLEMENTED';(out/'run.json').write_text(json.dumps(manifest,indent=2));return 1
if __name__=='__main__':sys.exit(main())
