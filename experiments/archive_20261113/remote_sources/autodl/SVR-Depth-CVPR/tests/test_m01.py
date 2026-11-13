"""CPU-only contract tests. These do not substitute for real CUDA inference."""
from __future__ import annotations
import contextlib
import csv
import hashlib
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from svr_depth.datasets.benchmark import iter_sequence
from svr_depth.evaluation.aggregate import paired_risk
from svr_depth.evaluation.m01_metrics import score_depth, score_ssi
from svr_depth.evaluation.m01_runner import execute, load_config
from svr_depth.streaming.m01_engine import M01Engine, _checkpoint, sha256_file


class MetricsTest(unittest.TestCase):
    def test_raw_scores_and_mask(self):
        gt = np.array([[1., 2.], [0., 4.]])
        pred = np.array([[2., 2.], [float('nan'), 2.]])
        mask = gt > 0
        r = score_depth(pred, gt, mask)
        self.assertEqual(r['status'], 'ok')
        self.assertAlmostEqual(r['absrel'], .5)
        self.assertAlmostEqual(r['rmse'], np.sqrt(5/3))
        self.assertAlmostEqual(r['delta1'], 1/3)
        self.assertEqual(score_depth(np.array([[0., 2.], [1., 2.]]), gt, mask)['status'], 'invalid_prediction')
        with self.assertRaises(ValueError):
            score_depth(pred, gt, np.ones((2, 2), dtype=bool))

    def test_ssi_is_optional_and_separate(self):
        gt = np.arange(1., 13.).reshape(3, 4)
        pred = gt * 2 + np.linspace(0, .2, 12).reshape(3, 4)
        raw = score_depth(pred, gt, np.ones_like(gt, dtype=bool))
        supplementary = score_ssi(pred, gt, np.ones_like(gt, dtype=bool))
        self.assertGreater(raw['absrel'], .9)
        self.assertEqual(supplementary['ssi_status'], 'ok')
        self.assertLess(supplementary['ssi_absrel'], .01)
        self.assertEqual(score_ssi(gt,gt,np.ones_like(gt,dtype=bool))['ssi_status'], 'insufficient_fit_pixels')

    def test_risk_never_compares_different_masks(self):
        base = {'dataset':'Bonn','scene':'s','sequence':'q','index':0,'seed':'x',
                'status':'ok','absrel':.1, 'rgb_sha256':'a', 'gt_sha256':'b', 'mask_sha256':'c'}
        with self.assertRaisesRegex(ValueError, 'mask_sha256'):
            paired_risk([{**base,'method':'A'}, {**base,'method':'P_multi','mask_sha256':'d'}])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch and upstream dependencies')
    def test_upstream_depth_metric_uses_original_functions(self):
        import torch
        from velodepth.utils.evaluation_depth import ssi, delta
        from svr_depth.evaluation.upstream_depth_metrics import score_upstream_depth
        gt = np.arange(1., 13., dtype=np.float32).reshape(3, 4)
        for pred in (gt.copy(), gt * 1.2 + .1):
            result = score_upstream_depth(gt, pred, np.ones_like(gt, dtype=bool))
            g = torch.from_numpy(gt.ravel()); p = torch.from_numpy(pred.ravel())
            self.assertEqual(result['upstream_depth_status'], 'ok')
            self.assertEqual(result['upstream_depth_d1_ssi'], float(delta(g, ssi(g, p), 1.0)))
        perfect = score_upstream_depth(gt, gt, np.ones_like(gt, dtype=bool))
        self.assertEqual(perfect['upstream_ssi_nonpositive_fraction'], 1.0)


class FakeEngine:
    """Predict simple constants while exposing the same runner contract."""
    def __init__(self, snapshot, method, base_sha):
        self.method = method
        self.identity = {'training_provenance_verified': False}
        self.reset()
    def reset(self):
        self.counts = {'keyframe_calls':0, 'flow_calls':0, 'native_fast_calls':0,
                       'repair_head_calls':0, 'output_adapter_calls':0, 'additional_readout_passes':0}
    def step(self, index, rgb):
        key = index % 16 == 0
        if key:
            self.counts['keyframe_calls'] += 1
        else:
            self.counts['flow_calls'] += 1
            self.counts['native_fast_calls'] += 1
            if self.method['method'] == 'P_multi': self.counts['repair_head_calls'] += 1
            if self.method['method'].startswith('O_'): self.counts['output_adapter_calls'] += 1
            if self.method['method'] == 'O_multi': self.counts['additional_readout_passes'] += 1
        value = {'A':1.0, 'P_multi':.9, 'O_fast':.95, 'O_multi':.98}[self.method['method']]
        return np.full(rgb.shape[:2], 1 if key else value), {'keyframe':key, 'age':index % 16}
    def synchronize(self): pass
    def to_numpy(self, pred): return pred
    def stats(self): return self.counts.copy()
    def peak_bytes(self): return 0


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.snapshot = self.root/'snapshot'
        self.snapshot.mkdir()
        (self.snapshot/'config.json').write_text('{}')
        (self.snapshot/'model.safetensors').write_bytes(b'synthetic-not-real-weights')
        methods = [{'method':'A','seed':'native'}]
        for method in ('P_multi','O_fast','O_multi'):
            ck = self.root/(method+'.pt')
            ck.write_bytes(b'synthetic-not-real-checkpoint')
            methods.append({'method':method, 'seed':'7', 'checkpoint':str(ck),
                            'sha256':sha256_file(ck), 'state_key':'head_state'})
        seqs = []
        for num in (1, 2):
            folder = self.root/f'scene{num}'
            folder.mkdir()
            rows=[]
            for i in range(18):
                rgb = folder/f'rgb{i:03d}.png'
                Image.fromarray(np.full((3, 4, 3), i, dtype=np.uint8)).save(rgb)
                depth = folder/f'depth{i:03d}.png'
                Image.fromarray(np.full((3, 4), 1000, dtype=np.uint16)).save(depth)
                unmatched = i == 1
                rows.append({'frame_index_zero_based':i, 'rgb_timestamp':i/30, 'rgb_path':str(rgb),
                             'gt_match_status':'UNMATCHED' if unmatched else 'MATCHED',
                             'depth_path':'' if unmatched else str(depth),
                             'depth_timestamp':'' if unmatched else i/30,
                             'abs_delta_ms':'' if unmatched else '0', 'gt_reused':'false'})
            csvpath = folder/'pairs.csv'
            with csvpath.open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            seqs.append({'dataset':'Bonn','sequence':f'seq{num}','scene':f'scene{num}',
                         'pairs_csv':str(csvpath),'depth_scale':1000,'min_depth_m':.001,
                         'max_depth_m':10,'max_delta_ms':20})
        self.cfg = {'refresh_interval':16,'gain':1,'frequency':1,
                    'snapshot':{'path':str(self.snapshot),'config_sha256':sha256_file(self.snapshot/'config.json'),
                                'weights_sha256':sha256_file(self.snapshot/'model.safetensors')},
                    'methods':methods,'sequences':seqs,'save_predictions':True,'bootstrap_samples':40,
                    'timing_repeats':3,'timing_warmup_frames':16}
        self.cfg_path=self.root/'config.json'
        self.cfg_path.write_text(json.dumps(self.cfg))
    def tearDown(self): self.tmp.cleanup()
    def test_complete_four_methods_with_unmatched_rgb_and_separate_timing(self):
        out=self.root/'quality'
        result=execute(self.cfg_path,out,'quality',engine_factory=FakeEngine)
        self.assertTrue(result['complete'])
        with (out/'frame_metrics.csv').open(encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows), 18*2*4)
        self.assertEqual(sum(r['status']=='unmatched' for r in rows),8)
        self.assertEqual(sum(r['keyframe']=='True' for r in rows),2*2*4)
        self.assertEqual(len(list((out/'predictions').rglob('*.npz'))), 18*2*4)
        with (out/'risk_summary.csv').open(encoding='utf-8-sig') as f: risk=list(csv.DictReader(f))
        self.assertEqual(len(risk), 8)
        self.assertEqual({int(r['paired_valid_frames']) for r in risk}, {17})
        with (out/'call_counts.csv').open(encoding='utf-8-sig') as f: counts=list(csv.DictReader(f))
        self.assertEqual({int(r['native_fast_calls']) for r in counts},{16})
        summary=json.loads((out/'dataset_summary.json').read_text())['datasets']
        self.assertTrue(all(x['ci_available'] for x in summary))
        timing=self.root/'timing'
        timing_result=execute(self.cfg_path,timing,'timing',engine_factory=FakeEngine)
        self.assertTrue(timing_result['complete'])
        with (timing/'latency_summary.csv').open(encoding='utf-8-sig') as f: timed=list(csv.DictReader(f))
        self.assertEqual(len(timed), 3*2*4)
        self.assertTrue(all(int(r['timed_frames'])==2 for r in timed))
        import subprocess, sys
        report = self.root/'report'
        script = Path(__file__).resolve().parents[1]/'scripts/m01_report.py'
        completed = subprocess.run([sys.executable,str(script),'--run-dir',str(out),'--out-dir',str(report),
                                    '--bootstrap-samples','10'], capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertTrue((report/'paired_comparison.json').is_file())
        with self.assertRaises(FileExistsError): execute(self.cfg_path,out,'quality',engine_factory=FakeEngine)
    def test_short_development_run_is_labeled_and_has_no_ci(self):
        out=self.root/'dev'
        result=execute(self.cfg_path,out,'quality',engine_factory=FakeEngine,development_frames=7)
        self.assertTrue(result['complete'])
        identity=json.loads((out/'run_identity.json').read_text())
        self.assertEqual(identity['run_scope'],'development_only_truncated')
        summary=json.loads((out/'dataset_summary.json').read_text())
        self.assertTrue(summary['development_only'])
        self.assertTrue(all(not row['ci_available'] for row in summary['datasets']))
        with (out/'frame_metrics.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        self.assertEqual(len(rows), 7*2*4)

    def test_inference_failure_marks_run_incomplete(self):
        class BrokenEngine(FakeEngine):
            def step(self,index,rgb):
                if self.method['method'] == 'P_multi' and index == 4:
                    raise RuntimeError('synthetic propagation failure')
                return super().step(index,rgb)
        out=self.root/'failed_run'
        result=execute(self.cfg_path,out,'quality',engine_factory=BrokenEngine)
        self.assertFalse(result['complete'])
        identity=json.loads((out/'run_identity.json').read_text())
        self.assertEqual(identity['status'],'incomplete')
        partial=json.loads((out/'dataset_summary.json').read_text())
        self.assertEqual(partial['source_run_status'],'incomplete')
        self.assertTrue(all(not row['ci_available'] for row in partial['datasets']))
        with (out/'failures.csv').open(encoding='utf-8-sig') as f: failures=list(csv.DictReader(f))
        self.assertEqual(len(failures),2)
        self.assertTrue(all('synthetic propagation failure' in r['reason'] for r in failures))

    def test_checkpoint_hash_rejected_before_output(self):
        self.cfg['methods'][1]['sha256']='0'*64
        self.cfg_path.write_text(json.dumps(self.cfg))
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
            execute(self.cfg_path,self.root/'bad','quality',engine_factory=FakeEngine)
        self.assertFalse((self.root/'bad').exists())

    def test_four_hdf5_datasets_require_unambiguous_source(self):
        for dataset in ('Bonn', 'TUM', 'ScanNet', 'Sintel'):
            with self.subTest(dataset=dataset):
                spec = {'dataset':dataset, 'sequence':'seq', 'scene':'s',
                        'hdf5':'data.hdf5', 'split_key':'train.txt',
                        'depth_scale':1000, 'min_depth_m':.001, 'max_depth_m':10}
                if dataset == 'ScanNet':
                    self.cfg['sequences'] = [dict(spec)]
                    self.cfg_path.write_text(json.dumps(self.cfg))
                    with self.assertRaisesRegex(ValueError, 'sequences_key'):
                        load_config(self.cfg_path)
                    spec['sequences_key'] = 'sequences_.json'
                self.cfg['sequences'] = [spec]
                self.cfg_path.write_text(json.dumps(self.cfg))
                actual = load_config(self.cfg_path)['sequences'][0]
                self.assertEqual(actual['hdf5'], str((self.root/'data.hdf5').resolve()))
                self.cfg['sequences'][0]['pairs_csv'] = 'also.csv'
                self.cfg_path.write_text(json.dumps(self.cfg))
                with self.assertRaisesRegex(ValueError, 'exactly one'):
                    load_config(self.cfg_path)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch and upstream dependencies')
    def test_upstream_depth_metric_is_recorded_and_summarized(self):
        self.cfg['methods'] = self.cfg['methods'][:1]
        self.cfg['sequences'] = self.cfg['sequences'][:1]
        self.cfg['compute_upstream_depth_metrics'] = True
        self.cfg_path.write_text(json.dumps(self.cfg))
        out = self.root/'upstream_metrics'
        result = execute(self.cfg_path, out, 'quality', engine_factory=FakeEngine, development_frames=3)
        self.assertTrue(result['complete'])
        with (out/'frame_metrics.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]['upstream_depth_status'], 'ok')
        with (out/'sequence_summary.csv').open(encoding='utf-8-sig') as f:
            summary = list(csv.DictReader(f))
        self.assertNotEqual(summary[0]['upstream_depth_d1_ssi'], '')

    def test_tum_hdf5_schema_with_in_memory_hdf5_stub(self):
        import io
        rgb = io.BytesIO(); Image.fromarray(np.full((3,4,3),20,dtype=np.uint8)).save(rgb, format='PNG')
        depth = io.BytesIO(); Image.fromarray(np.full((3,4),2000,dtype=np.uint16)).save(depth,format='PNG')
        sample={'image':['seq/im0','seq/im1'], 'depth':['seq/gt0',None],
                'camera_params':[[1]*16,[1]*16], 'camera_model':['OPENCV','OPENCV'],
                'cam2w':[None,None]}
        class Node:
            def __init__(self,value): self.value=value
            def __getitem__(self,key): return self.value
        class H5:
            def __init__(self,*a,**kw):
                self.data={'sequences.json':Node(json.dumps({'seq':sample}).encode()),
                           'val.txt':Node(b'seq 2\n'),
                           'seq/im0':Node(np.frombuffer(rgb.getvalue(),dtype=np.uint8)),
                           'seq/im1':Node(np.frombuffer(rgb.getvalue(),dtype=np.uint8)),
                           'seq/gt0':Node(np.frombuffer(depth.getvalue(),dtype=np.uint8))}
            def __enter__(self): return self
            def __exit__(self,*a): return False
            def __contains__(self,key): return key in self.data
            def __getitem__(self,key): return self.data[key]
        module=types.ModuleType('h5py'); module.File=H5
        fake_file=self.root/'tum.hdf5';fake_file.write_bytes(b'fake')
        spec={'dataset':'TUM','sequence':'seq','scene':'s','hdf5':str(fake_file),'split_key':'val.txt',
              'depth_scale':1000,'min_depth_m':.001,'max_depth_m':10}
        with patch.dict('sys.modules', {'h5py':module}):
            for dataset in ('Bonn', 'TUM', 'ScanNet', 'Sintel'):
                with self.subTest(dataset=dataset):
                    # The stub provides the same metadata under the explicit key.
                    selected = {**spec, 'dataset':dataset}
                    if dataset == 'ScanNet': selected['sequences_key'] = 'sequences.json'
                    rows=list(iter_sequence(selected))
                    self.assertEqual(len(rows),2)
                    self.assertTrue(np.all(rows[0].gt==2))
                    self.assertTrue(np.all(rows[0].gt_valid_mask))
                    self.assertIsNone(rows[1].gt)
                    self.assertEqual(rows[0].intrinsics['model'],'OPENCV')


class EngineWiringTest(unittest.TestCase):
    def test_uniform_fp32_flow_boundary_for_native_and_persistent(self):
        class Context:
            def __init__(self, state, value): self.state=state; self.value=value
            def __enter__(self): self.old=self.state.get('autocast'); self.state['autocast']=self.value
            def __exit__(self,*args): self.state['autocast']=self.old
        state={}
        fake_torch=types.SimpleNamespace(float16='fp16', no_grad=lambda:contextlib.nullcontext(),
            autocast=lambda *a,**kw: Context(state, kw.get('enabled',True)))
        class FakeStream:
            def __init__(self, rt, model, cfg, shape, adapter=None, debug=False):
                self.previous=None
            def advance(self, frame,index,static):
                if state['autocast'] is not True: raise AssertionError('inference must use autocast')
                self.previous=frame
                return 'pred', {'age':index, 'keyframe':index==0}
        flow_calls=[]
        def transition(rt,model,previous,frame,shape,dis):
            if state['autocast'] is not False: raise AssertionError('flow boundary must be FP32')
            flow_calls.append(1)
            return {'shared':'flow'}
        FakeStream.dis='DIS'
        rt=types.SimpleNamespace(torch=fake_torch, models=types.SimpleNamespace(Stream=FakeStream,transition=transition),
                                 gt2=types.SimpleNamespace(preprocess_rgb=lambda *a: ('img','frame_u8')))
        for family in ('A','P_multi'):
            engine=M01Engine.__new__(M01Engine)
            engine.rt=rt; engine.model=object();engine.method={'method':family};engine.device='cuda';engine.adapter=None
            engine.stream=None;engine.shape=None
            engine._shape_for=lambda rgb: {'original_shape':rgb.shape[:2], 'paddings':(0,0,0,0),
                                            'network_shape':rgb.shape[:2], 'mean':None,'std':None}
            rgb=np.zeros((4,5,3),dtype=np.uint8)
            engine.step(0,rgb); engine.step(1,rgb)
        self.assertEqual(len(flow_calls),2)

    def test_checkpoint_identity_and_smoke_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck=Path(tmp)/'head.pt';ck.write_bytes(b'synthetic')
            base='a'*64
            obj={'identity':{'method':'P_multi','seed':7,'snapshot_sha256':base}, 'head_state':{'weights':'fake'}}
            rt=types.SimpleNamespace(torch=types.SimpleNamespace(load=lambda *a,**kw:obj))
            spec={'method':'P_multi','seed':'7','checkpoint':str(ck),'sha256':sha256_file(ck),'state_key':'head_state'}
            state,meta=_checkpoint(rt,spec,base)
            self.assertEqual(state,{'weights':'fake'})
            self.assertTrue(meta['training_provenance_verified'])
            obj['identity']['method']='O_multi'
            with self.assertRaisesRegex(ValueError,'Checkpoint method'):
                _checkpoint(rt,spec,base)
            obj['identity']['method']='P_multi';obj['identity']['smoke']=True
            with self.assertRaisesRegex(ValueError,'Smoke checkpoint'):
                _checkpoint(rt,spec,base)


class StreamSemanticsTest(unittest.TestCase):
    def test_actual_historical_advance_schedule_and_output_state_invariant(self):
        import sys
        torch=types.ModuleType('torch')
        nn=types.ModuleType('torch.nn')
        functional=types.ModuleType('torch.nn.functional')
        nn.Module=type('Module',(),{})
        nn.functional=functional
        torch.nn=nn
        torch.float16='float16'
        torch.autocast=lambda *a,**kw: contextlib.nullcontext()
        source=Path(__file__).resolve().parents[1]/'svr_depth/_compat/output_smoke'
        sys.path.insert(0,str(source))
        try:
            with patch.dict('sys.modules', {'torch':torch,'torch.nn':nn,'torch.nn.functional':functional}):
                spec=importlib.util.spec_from_file_location('m01_test_y08_models', source/'y08y09_models.py')
                m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
                class T:
                    _version=0
                    def detach(self): return self
                m.output_depth=lambda *args:T()
                m.rgb_for_output=lambda *args:T()
                m.readout_radius=lambda *args:T()
                rt=types.SimpleNamespace(gt2=types.SimpleNamespace(make_z_depth=lambda *a:T()))
                for family in ('A','P','O_fast','O_multi'):
                    stream=m.Stream.__new__(m.Stream)
                    head=types.SimpleNamespace(calls=0)
                    stream.rt=rt;stream.model=types.SimpleNamespace(pixel_decoder=types.SimpleNamespace(persistent_feature_repair=head),interpolation_mode='bilinear')
                    stream.cfg={'family':family,'schedule':'K16','frequency':1,'gain':1.}
                    stream.shape={'padded_shape':(3,4),'paddings':(0,0,0,0)}
                    stream.debug=True
                    stream.adapter=(lambda *args:T()) if family=='O_fast' else (lambda features:list(features))
                    stream.reset()
                    def key(frame,index,auto_restart=False):
                        stream.state=(T(),T(),[T() for _ in range(4)],T());stream.last_key=index;stream.age=1
                        stream.counts['keyframe_calls']+=1
                    def fast(frame,static,apply):
                        if apply: head.calls+=1
                        stream.state=(T(),T(),[T() for _ in range(4)],T())
                        stream.counts['native_fast_calls']+=1
                        return None,None
                    stream.keyframe=key;stream.fast=fast
                    metas=[]
                    for idx in range(33):
                        _,meta=stream.advance({},idx,static={})
                        metas.append(meta)
                    self.assertEqual([i for i,x in enumerate(metas) if x['keyframe']], [0,16,32])
                    self.assertEqual([x['age'] for x in metas[:17]], list(range(16))+[0])
                    self.assertEqual(stream.counts['native_fast_calls'],30)
                    self.assertEqual(stream.counts['repair_head_calls'], 30 if family=='P' else 0)
                    self.assertEqual(stream.counts['output_adapter_calls'],30 if family.startswith('O_') else 0)
                    self.assertEqual(stream.counts['additional_readout_passes'],30 if family=='O_multi' else 0)
                # An output adapter must never mutate recursive state in-place.
                stream.cfg['family']='O_fast'
                def corrupt(*args):
                    stream.state[2][0]._version += 1
                    return T()
                stream.adapter=corrupt
                with self.assertRaisesRegex(RuntimeError,'modified recursive tensors'):
                    stream.advance({},33,static={})
        finally:
            sys.path.remove(str(source))


if __name__ == '__main__': unittest.main()
