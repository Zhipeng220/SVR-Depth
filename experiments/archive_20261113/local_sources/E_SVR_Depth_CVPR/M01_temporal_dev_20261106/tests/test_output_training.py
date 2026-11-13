import contextlib
import hashlib
import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'PyTorch required for gradient and propagation tests')
class OutputTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        sys.path.insert(0, str(ROOT / 'scripts'))
        import train_output_crosssource_v2 as subject
        cls.subject = subject

    def test_output_heads_update_without_recurrent_writeback(self):
        torch = self.torch
        for method in ('O_fast', 'O_multi'):
            with self.subTest(method=method):
                seen = []
                class Adapter(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.scale = torch.nn.Parameter(torch.tensor(1.0))
                    def forward(self, *args):
                        x = args[0]
                        return [t * self.scale for t in x] if isinstance(x, list) else x * self.scale
                def step(model, prev, curr, static, radius, confidence, features):
                    seen.append(float(features[0].item()))
                    return radius + 1, confidence, [features[0] + 1]
                gt2 = types.SimpleNamespace(make_z_depth=lambda r, *args: r,
                    masked_log_depth_loss=lambda p, g: (((p-g)**2).mean(), 1))
                models = types.SimpleNamespace(readout_radius=lambda model, f: f[0],
                                               rgb_for_output=lambda frame, shape: torch.tensor(0.0))
                rt = types.SimpleNamespace(gt2=gt2, models=models, exp1=types.SimpleNamespace(one_fast_step=step))
                model = types.SimpleNamespace(interpolation_mode='bilinear')
                frames = [{'gt': None}] + [{'gt': torch.tensor(0.0)} for _ in range(19)]
                window = {'frames': frames, 'statics': [{} for _ in range(19)],
                          'n_supervised': 19, 'padded_shape': (1, 1), 'paddings': (0,0,0,0),
                          'keyframe_state': {'radius': torch.tensor(1.0), 'confidence': torch.tensor(1.0),
                                             'features': [torch.tensor(0.0)], 'rays_map': torch.tensor(1.0)}}
                adapter = Adapter()
                optimizer = torch.optim.SGD(adapter.parameters(), lr=0.001)
                scaler = torch.amp.GradScaler('cuda', enabled=False)
                with mock.patch.object(torch, 'autocast', lambda *a, **kw: contextlib.nullcontext()):
                    result = self.subject.window_loss(rt, model, adapter, method, window, optimizer, scaler)
                self.assertEqual(seen, list(map(float, range(19))))
                self.assertNotEqual(adapter.scale.item(), 1.0)
                self.assertEqual(result['supervised_frames'], 19)
                self.assertEqual(window['keyframe_state']['features'][0].item(), 0.0)

    def test_m01_checkpoint_roundtrip_and_smoke_rejection(self):
        from svr_depth.streaming.m01_engine import _checkpoint
        torch = self.torch
        runtime = types.SimpleNamespace(torch=torch)
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'checkpoint.pt'
            base = 'a' * 64
            for method in ('O_fast', 'O_multi'):
                identity = {'method': method, 'seed': 20261103, 'snapshot_sha256': base, 'smoke': False}
                obj = {'identity': identity, 'adapter_state_dict': {'weight': torch.tensor([3.0])}}
                torch.save(obj, path)
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                spec = {'method': method, 'seed': 20261103, 'checkpoint': str(path),
                        'sha256': sha, 'state_key': 'adapter_state_dict'}
                state, info = _checkpoint(runtime, spec, base)
                self.assertEqual(state['weight'].item(), 3.0)
                self.assertTrue(info['training_provenance_verified'])
                obj['identity']['smoke'] = True
                torch.save(obj, path)
                spec['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, 'Smoke checkpoint'):
                    _checkpoint(runtime, spec, base)


if __name__ == '__main__':
    unittest.main()
