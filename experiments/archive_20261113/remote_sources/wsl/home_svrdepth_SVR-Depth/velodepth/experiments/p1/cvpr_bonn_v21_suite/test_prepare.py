import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from contract import digest
from prepare import freeze
from quality import verify
from test_contract import example

class PrepareTest(unittest.TestCase):
    def test_freeze_and_detect_changed_input(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);snap=root/'snapshot';snap.mkdir();(snap/'model.safetensors').write_bytes(b'base');(snap/'config.json').write_text('{}')
            rgb=root/'rgb.png';gt=root/'gt.png';rgb.write_bytes(b'rgb');gt.write_bytes(b'gt')
            manifest=root/'manifest.csv';r=dict(frame_index_zero_based=0,rgb_timestamp='1',rgb_path=str(rgb),depth_timestamp='1.001',depth_path=str(gt),gt_match_status='MATCHED',gt_reused='false')
            with manifest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerow(r)
            evidence=root/'evidence.json';evidence.write_text('{}');matched=root/'matched.json';c=example();c['snapshot']=str(snap);c['role']='development'
            for key in ('matched_training_evidence','selection_evidence','calibration_evidence','independence_evidence','m04_state_audit'):c[key]=str(evidence)
            c['sequences'][0].update(manifest=str(manifest),pairing_audit=str(evidence))
            for family,items in c['checkpoints'].items():
                for item in items:
                    p=root/f'{family}_{item["seed"]}.pt';p.write_text(family+str(item['seed']));item.update(path=str(p),sha256=digest(p),training_record=str(evidence))
            pairs=[]
            for i in range(3):
                r=dict(seed=i,P_checkpoint_sha256=c['checkpoints']['P'][i]['sha256'],O_multi_checkpoint_sha256=c['checkpoints']['O_multi'][i]['sha256'],P_steps=1000,O_multi_steps=1000,P_gt_exposures=50,O_multi_gt_exposures=50,checkpoint_rule='fixed_final_step')
                for field in ('initial_head_sha256','sample_order_sha256','supervision_sha256'):r['P_'+field]='a'*64;r['O_multi_'+field]='a'*64
                pairs.append(r)
            matched.write_text(json.dumps(dict(pairs=pairs)));c['matched_training_evidence']=str(matched)
            cfg=root/'config.json';cfg.write_text(json.dumps(c));out=root/'frozen.json'
            with patch('prepare.PINS',{}),patch('prepare.BASE_SHA',digest(snap/'model.safetensors')):freeze(cfg,out)
            p=json.loads(out.read_text());self.assertEqual(p['status'],'FROZEN_QUALITY_ONLY');self.assertEqual(len(p['units']),42)
            verify(p);gt.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Frozen input changed'):verify(p)
            with self.assertRaisesRegex(ValueError,'overwrite'):
                with patch('prepare.PINS',{}),patch('prepare.BASE_SHA',digest(snap/'model.safetensors')):freeze(cfg,out)

if __name__=='__main__':unittest.main()
