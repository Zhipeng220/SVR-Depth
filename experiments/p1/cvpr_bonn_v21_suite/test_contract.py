import csv,tempfile,unittest
from pathlib import Path
from contract import FAMILIES,validate_config,validate_manifest,reject_known_development_paths
from m04_branch import branch_step
from copy import deepcopy

def example():
    c=dict(version='BONN-V21-QUALITY-1',role='heldout',test_scores_unseen=True,calibration_audited=True,data_provenance_audited=True,same_head_state_audited=True,native_replay_rtol=1e-5,native_replay_atol=1e-4,depth_encoding='Bonn_registered_uint16_z_div5000',depth_min_m=.1,depth_max_m=10,selection_frozen_before_test=True,selection_budget_equal=True,candidate_counts=dict.fromkeys(FAMILIES,3),matched_training_evidence='e1',selection_evidence='e2',calibration_evidence='e3',independence_evidence='e4',m04_state_audit='e5',checkpoints={x:[dict(seed=i) for i in range(3)] for x in FAMILIES[1:]},
        sequences=[dict(sequence='new_bonn',physical_scene_group='space',physical_scene_verified=True,recording_session='one',sensor_model='s',camera_setup='c',manifest='m',audit_notes='checked',pairing_audit='p',pretraining_overlap='unknown')],points=[])
    for family in FAMILIES:
        for k in (16,32,64):
            c['points'].append(dict(id=f'{family}_{k}',family=family,k=k,gain=0 if family=='A' else 1,frequency=1,alignment='raw',role='primary' if family=='P' and k==16 else 'selected',semantics='documented'))
    return c

class ContractTests(unittest.TestCase):
    def test_unit_plan_and_no_relabeling(self):
        c=example();u=validate_config(c)
        self.assertEqual(len(u),42) # 30 M01/M02 units and 12 same-head M04 units
        self.assertEqual(len([x for x in u if x['experiment']=='M04']),12)
        c['sequences'][0]['sequence']='person_tracking2'
        with self.assertRaisesRegex(ValueError,'development'):validate_config(c)
        with self.assertRaisesRegex(ValueError,'development RGB path'):reject_known_development_paths([{'rgb_path':'/tmp/rgbd_bonn_crowd2/rgb/frame.png'}])
    def test_seed_pairing_and_thresholds(self):
        c=example();c['checkpoints']['O_multi'][0]['seed']=12
        with self.assertRaisesRegex(ValueError,'corresponding'):validate_config(c)
        c=example();c['depth_max_m']=0
        with self.assertRaisesRegex(ValueError,'range'):validate_config(c)
    def test_manifest_gt_reuse_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);gt=p/'gt';gt.write_text('x');rgb=[p/'r0',p/'r1']
            for x in rgb:x.write_text('x')
            f=p/'pairs.csv';rows=[dict(frame_index_zero_based=i,rgb_timestamp=1+i*.01,rgb_path=str(rgb[i]),depth_timestamp=1+i*.01,depth_path=str(gt),gt_match_status='MATCHED',gt_reused='false') for i in range(2)]
            with f.open('w',newline='') as o:
                w=csv.DictWriter(o,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            with self.assertRaisesRegex(ValueError,'GT path reused'):validate_manifest(f)
    def test_m04_same_parent_state(self):
        a=('r0','c0',['f0'],'ray');p=('r1','c1',['f1'],'ray')
        def step(repeats):return (a,'D0') if not repeats else (p,'D1')
        self.assertEqual(branch_step('OUTPUT_ONLY',1,16,step,deepcopy)[:2],(a,'D1'))
        self.assertEqual(branch_step('F_ONLY',1,16,step,deepcopy)[:2],(('r0','c0',['f1'],'ray'),'D0'))

if __name__=='__main__':unittest.main()
