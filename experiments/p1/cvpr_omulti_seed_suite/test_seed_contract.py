import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import seed_contract as c

def p_metadata(seed,pair_shas,init_sha):
    return dict(y04_outer_seed=seed,mode='pre_readout',y04_label='Y04_P_multi',y04_kind='persistent',step_count=1000,optimizer_steps=1000,
        pretrained_weights_sha256=c.BASE_SHA,y04_initial_head_sha256=init_sha,y04_sample_order_sha256=c.order_sha(seed),
        training_pair_csv_sha256_by_sequence=pair_shas,training_cycle_slots=[list(x) for x in c.SLOTS],training_sequences=['crowd2','crowd3','balloon2'],repair_head_state_dict={'weight':'fixture'})

class ContractTests(unittest.TestCase):
    def test_known_seed_order(self):
        for seed in c.SEEDS:self.assertEqual(c.order_sha(seed),c.ORDER_SHA[seed])
    def test_mismatched_seed_order_and_initialization_rejected(self):
        pairs=dict.fromkeys(('crowd2','crowd3','balloon2'),'m');x=p_metadata(20261101,pairs,'init')
        c.check_p_metadata(x,20261101,'hash',pairs,'init')
        for key,value in [('y04_outer_seed',20261102),('y04_sample_order_sha256','wrong'),('y04_initial_head_sha256','wrong')]:
            y=dict(x);y[key]=value
            with self.assertRaises(ValueError):c.check_p_metadata(y,20261101,'hash',pairs,'init')
    def test_training_inputs_detect_bytes_changed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);rgb=p/'rgb';gt=p/'gt';rgb.write_bytes(b'rgb');gt.write_bytes(b'gt')
            row={'rgb_path':str(rgb),'depth_path':str(gt),'gt_match_status':'MATCHED'}
            manifests={s:[row]*256 for s in ('crowd2','crowd3','balloon2')}
            before=c.training_inputs(manifests);gt.write_bytes(b'changed');after=c.training_inputs(manifests)
            self.assertNotEqual(before,after)
    def test_p_input_provenance_is_not_fabricated(self):
        x=c.incomplete_pair(20261101,'P','O','init','o-supervision',120)
        self.assertEqual(x['P_supervision_sha256'],'');self.assertEqual(x['O_multi_supervision_sha256'],'o-supervision')

if __name__=='__main__':unittest.main()
