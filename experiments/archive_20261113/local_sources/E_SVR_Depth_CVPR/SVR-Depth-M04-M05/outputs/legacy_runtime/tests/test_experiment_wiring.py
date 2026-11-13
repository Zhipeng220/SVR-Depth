"""Dependency-free checks of experiment source identity and M05 call accounting."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    path = ROOT / 'experiments' / 'p1' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class FakeModule:
    def register_forward_hook(self, hook):
        self.hook = hook
        return self
    def remove(self):
        pass

class ExperimentWiring(unittest.TestCase):
    def test_fresh_training_order_matches_m05_protocol(self):
        train, m05 = load('train_pmulti_fresh'), load('cvpr_m05_formal')
        for seed in train.SEEDS:
            self.assertEqual(train.balanced_order(seed), m05.y04_balanced_order(24, 1000, seed))
            self.assertEqual(len(train.balanced_order(seed)), 1000)

    def test_pinned_sources_match_historical_lock(self):
        m04, m05 = load('cvpr_m04_closed_loop'), load('cvpr_m05_formal')
        expected = {k: v['actual'] for k, v in json.loads(
            (ROOT / 'reference' / 'M05_protocol_lock.json').read_text(encoding='utf8')
        )['source_hashes'].items()}
        self.assertEqual(m04.source_hashes(), expected)
        self.assertEqual({k: v['actual'] for k, v in m05.verify_sources().items()}, expected)

    def test_m05_module_counter_records_calls(self):
        from types import SimpleNamespace
        m05 = load('cvpr_m05_formal')
        encoder, small, decoder = FakeModule(), FakeModule(), FakeModule()
        decoder.radial_module = FakeModule()
        decoder.persistent_feature_repair = FakeModule()
        model = SimpleNamespace(pixel_encoder=encoder, small_encoder=small, pixel_decoder=decoder)
        counter = m05.CallCounter(model)
        counter.set_stage('example')
        encoder.hook(None, None, None)
        encoder.hook(None, None, None)
        row = next(r for r in counter.rows() if r['module'] == 'pixel_encoder')
        self.assertEqual(row['calls'], 2)
        counter.close()
