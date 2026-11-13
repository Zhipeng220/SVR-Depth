import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from contract import digest
from init_config import create,DEVELOPMENT

class InitTest(unittest.TestCase):
    def test_discovery_does_not_mark_audits_passed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);dev=root/'outputs/gt4a2_exp1';dev.mkdir(parents=True)
            for s in ('crowd2','crowd3','balloon2'):(dev/f'gt4a2_exp1_{s}_pairs_used.csv').write_text('fixture')
            paths={}
            for s in DEVELOPMENT:
                p=root/(s+'.csv');p.write_text('fixture');paths[s]={'path':str(p)}
            dp=root/'outputs/y11_confirmation_data/dataset_protocol.json';dp.parent.mkdir(parents=True);dp.write_text(json.dumps({'manifests':paths}))
            out=root/'candidate.json'
            with patch('init_config.DATA_SHA',digest(dp)):create(root,out,'')
            c=json.loads(out.read_text());self.assertEqual(len(c['known_development_manifests']),8);self.assertFalse(c['calibration_audited']);self.assertEqual(c['sequences'],[])
            with self.assertRaisesRegex(ValueError,'overwrite'):create(root,out,'')

if __name__=='__main__':unittest.main()
