import importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock

HERE=Path(__file__).resolve().parent
class RunnerTests(unittest.TestCase):
 def test_smoke_seed_and_input_guard(self):
  for changed in (False,True):
   with self.subTest(changed=changed),tempfile.TemporaryDirectory() as td:
    root=Path(td);(root/'outputs').mkdir(); data=root/'input.bin';data.write_bytes(b'original'); lock=root/'gpu.lock';lock.touch();out=root/'outputs'/'run'
    fakefcntl=types.ModuleType('fcntl');fakefcntl.LOCK_EX=2;fakefcntl.flock=lambda *a:None
    y=types.ModuleType('gt4a2_y08cal_y09pareto'); y.SEQS=('crowd2','crowd3','balloon2');y.SEED=20261025
    def save(p,obj):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(obj),encoding='utf-8')
    y.json_write=save;y.json_read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    cuda=types.SimpleNamespace(is_available=lambda:True,get_device_properties=lambda _:types.SimpleNamespace(uuid='abc'))
    rt=types.SimpleNamespace(torch=types.SimpleNamespace(cuda=cuda,set_num_threads=lambda n:None))
    with mock.patch.dict(sys.modules,{'fcntl':fakefcntl}):
     spec=importlib.util.spec_from_file_location('ofast_runner_test',HERE/'train_ofast.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.PINS={};y.preflight=lambda a:({'input_sha256':{str(data):r.sha(data)}},{},{},rt,{}, {},None)
    y.seed=lambda rt:None;y.build_training_cache=lambda *a:{}
    def train(a,rt,family,paths,out,signature,pstate,ostate):
     assert family=='O_fast' and y.SEED==20261102
     path=out/'training/O_fast';path.mkdir(parents=True);(path/'final.pt').write_bytes(b'head');save(path/'summary.json',{'steps':3})
     if changed:data.write_bytes(b'changed')
     return {},{}
    y.train_head=train;gpu=types.ModuleType('gt4a2_y11_fallback');gpu.gpu_lock_path=lambda:('abc',lock)
    argv=['train_ofast.py','--seed','20261102','--out-dir',str(out),'--snapshot',str(root/'model'),'--smoke-only']
    with mock.patch.dict(sys.modules,{'gt4a2_y08cal_y09pareto':y,'gt4a2_y11_fallback':gpu}),mock.patch.object(Path,'cwd',return_value=root),mock.patch.object(sys,'argv',argv),mock.patch.object(r,'fcntl',fakefcntl):
     if changed:
      with self.assertRaisesRegex(RuntimeError,'input changed'):r.main()
      self.assertFalse((out/'m01_ofast_status.json').exists());self.assertTrue(list(out.glob('failure_*.json')))
     else:
      r.main();status=json.loads((out/'m01_ofast_status.json').read_text());self.assertEqual(status['status'],'SMOKE_PASS');self.assertEqual(status['seed'],20261102);self.assertFalse(status['evaluation_performed'])
      protocol=json.loads((out/'protocol.json').read_text());self.assertEqual(protocol['m01_ofast_training']['order_policy'],'Y09 fixed 20261025 order for all O_fast seeds')
if __name__=='__main__':unittest.main()
