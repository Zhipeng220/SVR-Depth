import unittest
from stream_factory import build_stream, verify_counts

class DummyStream:
    def __init__(self, rt, model, cfg, shape, adapter=None):
        self.cfg = cfg
        self.adapter = adapter

    def propagated_frame(self, features):
        if self.cfg['family'] == 'O_multi':
            return self.adapter(features)
        return features

class StreamFactoryTest(unittest.TestCase):
    def test_output_adapter_is_invoked_on_propagation(self):
        output_head = lambda features: features + 7
        stream = build_stream(DummyStream, None, None, None, 'O_multi', 16, output_head)
        self.assertEqual(stream.propagated_frame(3), 10)
        self.assertEqual(stream.cfg['schedule'], 'K16')

    def test_missing_or_unexpected_adapter_rejected(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            build_stream(DummyStream, None, None, None, 'O_multi', 16, None)
        with self.assertRaisesRegex(ValueError, 'must not'):
            build_stream(DummyStream, None, None, None, 'A', 16, object())
        self.assertEqual(build_stream(DummyStream, None, None, None, 'P', 32, None).cfg['schedule'], 'K32')

    def test_output_adapter_call_ledger(self):
        counts = dict(keyframe_calls=3, flow_calls=37, native_fast_calls=37,
                      repair_head_calls=0, output_adapter_calls=37,
                      additional_readout_passes=37, rejected_candidates=0)
        verify_counts(counts, 'O_multi', 16, 40)
        counts['output_adapter_calls'] = 0
        with self.assertRaisesRegex(RuntimeError, 'call counts'):
            verify_counts(counts, 'O_multi', 16, 40)

if __name__ == '__main__': unittest.main()
