import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'experiments/risk_update/oracle_probe.py'
spec = importlib.util.spec_from_file_location('risk_oracle', PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture():
    common=dict(seed='1',sequence='scene',cycle='0',age='4',parent='A_native',anchor_index='4')
    states=[]; current=[]; future=[]
    for b in 'AOFT':
        s=dict(common,branch=b,parent_state_sha256='parent',committed_state_sha256='old' if b in 'AO' else 'new')
        states.append(s)
        if b in 'OT':
            current.append(dict(s,current_output_source='D1',gt_matched='True',AbsRel='0.1',RMSE='0.2',delta1='0.9',valid_pixels='100',total_pixels='100'))
            for h in range(1,5):
                future.append(dict(common,branch=b,horizon=str(h),frame_index=str(4+h),repair_mode_after_intervention='off',static_cache_unchanged='True',gt_matched='True',valid_pixels='100',total_pixels='100',AbsRel='0.12' if b=='O' else '0.10'))
    return future, states, current, {'all_invariants_pass':True,'invariant_checks':1,'K':16,'H':4}


class OracleTests(unittest.TestCase):
    def test_paired_min_and_horizons(self):
        f,s,c,z=fixture(); records,seq,macro=mod.analyze(f,s,c,z)
        self.assertEqual(len(records),4)
        self.assertAlmostEqual(records[3]['delta_T_minus_O'],-0.02)
        self.assertEqual(records[3]['local_oracle'],0.1)
        self.assertEqual(records[3]['choice'],'T')
        self.assertEqual(seq[3]['valid'],1)
        self.assertEqual(macro[3]['sequences'],1)

    def test_missing_gt_not_zero(self):
        f,s,c,z=fixture(); f[3]['gt_matched']='False'
        records,_,_=mod.analyze(f,s,c,z)
        self.assertFalse(records[3]['valid'])
        self.assertIsNone(records[3]['local_oracle'])

    def test_mismatched_parent_fails(self):
        f,s,c,z=fixture(); s[1]['parent_state_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'parent mismatch'): mod.analyze(f,s,c,z)

    def test_invalid_replay_fails(self):
        f,s,c,z=fixture(); f[0]['repair_mode_after_intervention']='pre_readout'
        with self.assertRaisesRegex(ValueError,'replay contract'): mod.analyze(f,s,c,z)

    def test_mismatched_pixels_fail(self):
        f,s,c,z=fixture(); f[4]['valid_pixels']='99'
        with self.assertRaisesRegex(ValueError,'pixel-count mismatch'): mod.analyze(f,s,c,z)


if __name__ == '__main__':
    unittest.main()
