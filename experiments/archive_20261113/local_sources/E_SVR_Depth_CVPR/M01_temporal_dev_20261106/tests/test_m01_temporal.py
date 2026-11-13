"""Contract tests for provisional temporal reporting and explicit metadata."""
from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
import csv
import json
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from svr_depth.datasets.benchmark import FrameRecord, decode_benchmark_flow, iter_sequence
from svr_depth.evaluation.m01_temporal import score_temporal_pair
from svr_depth.evaluation.paper_aggregate import summarize_coverage, require_formal_aggregate
from svr_depth.evaluation.m01_runner import execute
from svr_depth.streaming.m01_engine import sha256_file


def frame(index, **changes):
    h, w = 3, 4
    gt = np.arange(1., h*w + 1).reshape(h, w).astype(np.float32)
    zero = np.zeros((h, w, 2), dtype=np.float32)
    mask = np.ones((h, w), dtype=bool)
    base = FrameRecord('TUM', 'seq', 'scene', index, np.zeros((h,w,3), np.uint8), gt, float(index),
                       intrinsics=np.array([[3.,0,1.5],[0,3.,1],[0,0,1.]]), pose=np.eye(4),
                       gt_valid_mask=mask.copy(), flow_fwd=zero.copy(), flow_bwd=zero.copy(),
                       flow_fwd_mask=mask.copy(), flow_bwd_mask=mask.copy(), pose_valid=True,
                       camera_valid=True, flow_valid=True, source_frame_id=f'img{index}', source_frame_index=index,
                       flow_fwd_source=f'fwd{index}', flow_bwd_source=f'bwd{index}',
                       flow_available_fields=('flow_fwd','flow_bwd','flow_fwd_mask','flow_bwd_mask'))
    return replace(base, **changes)


class TemporalContractTests(unittest.TestCase):
    def test_identity_pair_uses_supplied_ssi_and_is_provisional(self):
        a,b=frame(0),frame(1)
        result=score_temporal_pair({'frame':a,'prediction':a.gt.copy(),'ssi_aligned':a.gt.copy()},
                                   {'frame':b,'prediction':b.gt.copy(),'ssi_aligned':b.gt.copy()},
                                   {'protocol':'provisional_v1','fb_max_error_px':.25})
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['protocol_status'],'provisional')
        self.assertEqual(result['tau5_provisional'],1.)
        self.assertEqual(result['tau5_ssi_provisional'],1.)
        self.assertEqual(result['valid_pixels'],12)

    def test_missing_flow_unverified_camera_and_nonconsecutive_are_explicit(self):
        a=frame(0)
        b=frame(2, flow_bwd=None, flow_available_fields=('flow_fwd',),
                camera_valid=False, pose_valid=False, pose=None)
        result=score_temporal_pair({'frame':a,'prediction':a.gt}, {'frame':b,'prediction':b.gt},
                                   {'protocol':'provisional_v1'})
        for reason in ('nonconsecutive_frames','missing_backward_flow','unverified_camera',
                       'missing_pose','unconfigured_fb_threshold'):
            self.assertIn(reason,result['issues'])
        self.assertIsNone(result['tau5_provisional'])
        with self.assertRaisesRegex(ValueError, 'provisional_v1'):
            score_temporal_pair({'frame':a},{'frame':b},{'protocol':'verified'})

    def test_missing_ssi_keeps_raw_temporal_result(self):
        a,b=frame(0),frame(1)
        r=score_temporal_pair({'frame':a,'prediction':a.gt}, {'frame':b,'prediction':b.gt},
                              {'fb_max_error_px':.25})
        self.assertEqual(r['tau5_provisional'],1.)
        self.assertEqual(r['ssi_status'],'ssi_unavailable')
        self.assertIsNone(r['tau5_ssi_provisional'])

    def test_temporal_outputs_include_flow_identity(self):
        a,b=frame(0),frame(1)
        r=score_temporal_pair({'frame':a,'prediction':a.gt,'ssi_aligned':a.gt},
                              {'frame':b,'prediction':b.gt,'ssi_aligned':b.gt},
                              {'fb_max_error_px':.25})
        self.assertEqual(r['flow_fwd_source'],'fwd0')
        self.assertEqual(r['flow_bwd_source'],'bwd1')
        self.assertEqual(len(r['flow_fwd_sha256']),64)
        self.assertEqual(len(r['flow_bwd_mask_sha256']),64)
        b=replace(b,flow_valid=False)
        r=score_temporal_pair({'frame':a,'prediction':a.gt}, {'frame':b,'prediction':b.gt},
                              {'fb_max_error_px':.25})
        self.assertIn('unverified_flow_contract',r['issues'])

    def test_invalid_camera_pose_and_flow_mask_do_not_get_scores(self):
        a=frame(0)
        cases=[frame(1, intrinsics={'model':'OPENCV'}), frame(1, pose=np.zeros((4,4))),
               frame(1, flow_bwd_mask=None)]
        statuses=['unsupported_or_unverified_camera_model','invalid_temporal_input','missing_backward_mask']
        for b,status in zip(cases,statuses):
            with self.subTest(status=status):
                r=score_temporal_pair({'frame':a,'prediction':a.gt}, {'frame':b,'prediction':b.gt},
                                      {'fb_max_error_px':.25})
                self.assertEqual(r['status'],status)
                self.assertIsNone(r['tau5_provisional'])

    def test_packed_flow_requires_explicit_geometry(self):
        # 2047 in each 12-bit channel maps to exactly zero in upstream decoder.
        packed=np.empty((3,4,3),dtype=np.uint8); packed[:]=[255,247,127]
        buf=BytesIO();Image.fromarray(packed).save(buf,format='PNG')
        flow=decode_benchmark_flow(buf.getvalue(), encoding='velodepth_packed12_rgb',
                                   geometry='fraction_of_extent')
        np.testing.assert_allclose(flow,0,atol=1e-7)
        with self.assertRaisesRegex(ValueError,'unverified'):
            decode_benchmark_flow(buf.getvalue(), encoding='velodepth_packed12_rgb', geometry='unknown')

    def test_hdf5_flow_offsets_are_explicit_not_guessed(self):
        def png(a):
            b=BytesIO();Image.fromarray(a).save(b,format='PNG');return b.getvalue()
        rgb=png(np.zeros((3,4,3),np.uint8)); depth=png(np.full((3,4),2000,np.uint16))
        packed=np.empty((3,4,3),np.uint8);packed[:]=[255,247,127]
        flow=png(packed);mask=png(np.full((3,4),255,np.uint8))
        sample={'image':['im0','im1'],'depth':['gt0','gt1'],
                'K':[np.eye(3).tolist()]*2,'cam2w':[np.eye(4).tolist()]*2,
                'flow_fwd':[None,'fwd0','fwd1'],'flow_fwd_mask':[None,'fm0','fm1'],
                'flow_bwd':[None,'bwd1'],'flow_bwd_mask':[None,'bm1']}
        class Node:
            def __init__(self,v):self.v=v
            def __getitem__(self,k):return self.v
        class H5:
            def __init__(self,*a,**k):
                self.data={'sequences.json':Node(json.dumps({'seq':sample}).encode()),'train.txt':Node(b'seq 2\n')}
                for n,v in {'im0':rgb,'im1':rgb,'gt0':depth,'gt1':depth,'fwd0':flow,'fwd1':flow,
                            'bwd1':flow,'fm0':mask,'fm1':mask,'bm1':mask}.items():
                    self.data[n]=Node(np.frombuffer(v,dtype=np.uint8))
            def __enter__(self):return self
            def __exit__(self,*a):return False
            def __contains__(self,k):return k in self.data
            def __getitem__(self,k):return self.data[k]
        h5py=types.ModuleType('h5py');h5py.File=H5
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.hdf5';p.write_bytes(b'stub')
            spec={'dataset':'TUM','sequence':'seq','scene':'s','hdf5':str(p),'split_key':'train.txt',
                  'depth_scale':1000,'min_depth_m':.001,'max_depth_m':10}
            with patch.dict('sys.modules',{'h5py':h5py}):
                no_contract=list(iter_sequence(spec))
                self.assertIsNone(no_contract[0].flow_fwd)
                self.assertEqual(no_contract[0].flow_error,'unverified_flow_contract')
                from svr_depth.evaluation.m01_data_audit import inspect_hdf5
                audit=inspect_hdf5(spec, max_frames=2)
                self.assertEqual(audit['field_lengths']['flow_fwd'],3)
                self.assertEqual(audit['pairs'][0]['pairing_status'],'unverified_metadata_candidates_only')
                spec['temporal_inputs']={'flow_contract_verified':True,'flow_evidence':'synthetic',
                    'flow_encoding':'velodepth_packed12_rgb','flow_geometry':'fraction_of_extent',
                    'fwd_field':'flow_fwd','fwd_mask_field':'flow_fwd_mask','fwd_index_offset':1,
                    'bwd_field':'flow_bwd','bwd_mask_field':'flow_bwd_mask','bwd_index_offset':0,
                    'pose_provenance_verified':True,'pose_evidence':'synthetic',
                    'camera_geometry_verified':True,'camera_evidence':'synthetic'}
                rows=list(iter_sequence(spec))
            self.assertEqual(rows[0].flow_fwd_source,'fwd0')
            self.assertEqual(rows[1].flow_bwd_source,'bwd1')
            self.assertTrue(rows[0].flow_fwd_mask.all())
            self.assertTrue(rows[1].flow_bwd_mask.all())
            self.assertTrue(rows[0].pose_valid and rows[0].camera_valid)
            self.assertTrue(rows[0].flow_valid)
            np.testing.assert_allclose(rows[0].flow_fwd,0,atol=1e-7)

    def test_official_aggregate_stays_blocked(self):
        report=summarize_coverage([],[],development=True)
        self.assertFalse(report['formal_aggregate_ready'])
        self.assertIsNone(report['formal_aggregate'])
        self.assertIn('author_temporal_equivalence_unverified',report['blocking_reasons'])
        with self.assertRaisesRegex(RuntimeError,'unavailable'):
            require_formal_aggregate()

    def test_upstream_depth_weighting_is_diagnostic_only(self):
        rows=[]
        for n,ds in enumerate(('Bonn','TUM','ScanNet','Sintel'), start=1):
            rows.append({'dataset':ds,'method':'A','seed':'native','sequence':f'seq{n}',
                         'status':'ok','valid_pixels':100,'upstream_depth_status':'ok',
                         'upstream_ssi_fit_status':'ok','upstream_depth_d1':n/10,
                         'upstream_depth_d1_ssi':n/20})
        report=summarize_coverage(rows,[],development=True)
        self.assertFalse(report['formal_aggregate_ready'])
        self.assertAlmostEqual(report['upstream_depth_equal_dataset_diagnostic'][0]['raw_d1'],.25)
        self.assertAlmostEqual(report['upstream_depth_equal_dataset_diagnostic'][0]['d1_ssi'],.125)
        self.assertEqual(summarize_coverage(rows[:-1],[])['upstream_depth_equal_dataset_diagnostic'],[])


class TemporalRunnerTests(unittest.TestCase):
    def test_adjacent_missing_gt_and_flow_are_not_inference_failures(self):
        from test_m01 import FakeEngine
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);snap=root/'snapshot';snap.mkdir()
            (snap/'config.json').write_text('{}');(snap/'model.safetensors').write_bytes(b'test')
            csv_path=root/'pairs.csv';csv_path.write_text('synthetic manifest')
            cfg={'refresh_interval':16,'gain':1,'frequency':1,
                 'snapshot':{'path':str(snap),'config_sha256':sha256_file(snap/'config.json'),
                             'weights_sha256':sha256_file(snap/'model.safetensors')},
                 'methods':[{'method':'A','seed':'native'}],
                 'sequences':[{'dataset':'Bonn','sequence':'seq','scene':'scene',
                               'pairs_csv':str(csv_path),'max_delta_ms':20,'depth_scale':1000,
                               'min_depth_m':.001,'max_depth_m':10}],
                 'temporal':{'enabled':True,'protocol':'provisional_v1'},'bootstrap_samples':0}
            cfg_path=root/'config.json';cfg_path.write_text(json.dumps(cfg))
            frames=[frame(0,flow_fwd=None,flow_bwd=None,flow_available_fields=()),
                    frame(1,gt=None,gt_valid_mask=None,flow_fwd=None,flow_bwd=None,flow_available_fields=()),
                    frame(2,flow_fwd=None,flow_bwd=None,flow_available_fields=())]
            out=root/'out'
            with patch('svr_depth.evaluation.m01_runner.iter_sequence',return_value=iter(frames)):
                # Default iterator argument is bound at definition time: inject at run_quality instead.
                from svr_depth.evaluation.m01_runner import run_quality
                out.mkdir()
                result=run_quality(cfg,out,FakeEngine,'fake',iterator_factory=lambda spec:iter(frames))
            self.assertTrue(result['complete'])
            self.assertEqual(result['frame_rows'],3)
            self.assertEqual(result['temporal_rows'],2)
            with (out/'temporal_metrics.csv').open(encoding='utf-8-sig') as f: pairs=list(csv.DictReader(f))
            self.assertEqual(len(pairs),2)
            self.assertTrue(all('missing_gt' in x['issues'] and 'missing_backward_flow' in x['issues'] for x in pairs))
            with (out/'failures.csv').open(encoding='utf-8-sig') as f: failures=list(csv.DictReader(f))
            self.assertFalse(any(r.get('stage')=='inference' for r in failures))
            report=json.loads((out/'paper_coverage.json').read_text())
            self.assertFalse(report['formal_aggregate_ready'])

    def test_metric_exception_does_not_stop_model_sequence(self):
        from test_m01 import FakeEngine
        from svr_depth.evaluation.m01_runner import run_quality
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir()
            cfg={'methods':[{'method':'A','seed':'native'}],
                 'snapshot':{'path':str(root)},
                 'sequences':[{'dataset':'TUM','sequence':'seq','scene':'scene'}],
                 'temporal':{'enabled':True,'protocol':'provisional_v1'},'bootstrap_samples':0}
            records=[frame(i,flow_fwd=None,flow_bwd=None,flow_available_fields=()) for i in range(3)]
            from svr_depth.evaluation.m01_metrics import score_depth as real_score
            count={'value':0}
            def score(*args):
                count['value']+=1
                if count['value']==1:raise ValueError('synthetic score failure')
                return real_score(*args)
            with patch('svr_depth.evaluation.m01_runner.score_depth',side_effect=score):
                result=run_quality(cfg,out,FakeEngine,'fake',iterator_factory=lambda spec:iter(records))
            self.assertFalse(result['complete'])
            self.assertEqual(result['frame_rows'],3)
            self.assertEqual(result['temporal_rows'],2)
            with (out/'failures.csv').open(encoding='utf-8-sig') as f:failures=list(csv.DictReader(f))
            self.assertEqual(failures[0]['stage'],'metric_or_output')

    def test_valid_synthetic_pairs_are_written_as_provisional(self):
        from test_m01 import FakeEngine
        from svr_depth.evaluation.m01_runner import run_quality
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir()
            cfg={'methods':[{'method':'A','seed':'native'}], 'snapshot':{'path':str(root)},
                 'sequences':[{'dataset':'TUM','sequence':'seq','scene':'scene'}],
                 'temporal':{'enabled':True,'protocol':'provisional_v1','fb_max_error_px':.25},
                 'bootstrap_samples':0}
            records=[frame(i) for i in range(3)]
            result=run_quality(cfg,out,FakeEngine,'fake',iterator_factory=lambda spec:iter(records))
            self.assertTrue(result['complete'])
            with (out/'temporal_metrics.csv').open(encoding='utf-8-sig') as f:pairs=list(csv.DictReader(f))
            self.assertEqual(len(pairs),2)
            self.assertTrue(all(r['status']=='ok' and r['protocol_status']=='provisional' for r in pairs))
            self.assertTrue(all(r['ssi_status']=='ssi_unavailable' for r in pairs))
            self.assertTrue(all(float(r['tau5_provisional'])==1 for r in pairs))


if __name__ == '__main__':
    unittest.main()
