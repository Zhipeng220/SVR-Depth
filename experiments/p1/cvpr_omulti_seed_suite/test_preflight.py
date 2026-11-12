import json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import seed_contract as contract
import train_omulti as train
from test_seed_contract import p_metadata

class PreflightTest(unittest.TestCase):
    def test_preflight_without_cuda(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'velodepth/models').mkdir(parents=True);(root/'outputs').mkdir()
            snap=root/'snap';snap.mkdir();(snap/'config.json').write_text('{}');(snap/'model.safetensors').write_bytes(b'base');base=contract.sha(snap/'model.safetensors')
            pcp=root/'P.pt';pcp.write_bytes(b'P');p_sha=contract.sha(pcp);ps=root/'P_summary.json';ps.write_text(json.dumps({'checkpoint_sha256':p_sha,'y04_outer_seed':20261101,'y04_sample_order_sha256':contract.order_sha(20261101)}))
            rgb=root/'rgb';gt=root/'gt';rgb.write_bytes(b'rgb');gt.write_bytes(b'gt')
            paths={};manifests={}
            for seq in ('crowd2','crowd3','balloon2'):
                paths[seq]=root/f'{seq}.csv';paths[seq].write_text(seq)
                manifests[seq]=[dict(rgb_path=str(rgb),depth_path=str(gt),gt_match_status='MATCHED') for _ in range(256)]
            pair_shas={k:contract.sha(v) for k,v in paths.items()};obj=p_metadata(20261101,pair_shas,'init');obj['pretrained_weights_sha256']=base
            context=dict(initial_head_hash='init',initial_head_state={});q03=types.SimpleNamespace(build_preflight=lambda *a:({},manifests,None,None,None,context))
            def identity(path,value,resume=False):Path(path).write_text(json.dumps(value))
            fake=types.SimpleNamespace(SEQUENCES=('crowd2','crowd3','balloon2'),load_runtime=lambda:None,q03=q03,torch=types.SimpleNamespace(load=lambda *a,**k:obj),args_for_q03=lambda x:x,balanced_order=lambda n,steps,seed:contract.order(seed),ensure_run_identity=identity)
            argv=['--seed','20261101','--snapshot',str(snap),'--p-checkpoint',str(pcp),'--p-sha256',p_sha,'--p-summary',str(ps),'--out-dir',str(root/'outputs/cvpr_omulti_seed_20261101'),'--preflight-only']
            for seq,path in paths.items():argv.extend(['--'+seq+'-pair-csv',str(path)])
            args=train.parser().parse_args(argv)
            with patch('train_omulti.Path.cwd',return_value=root),patch.object(train,'PINS',{}),patch.object(train,'BASE_SHA',base),patch.object(contract,'BASE_SHA',base),patch.object(train.importlib,'import_module',return_value=fake):
                self.assertEqual(train.run(args),0)
            report=json.loads((root/'outputs/cvpr_omulti_seed_20261101/preflight.json').read_text())
            self.assertEqual(report['status'],'PREFLIGHT_PASS');self.assertFalse(report['training_performed']);self.assertEqual(report['expected_GT_exposures'],31000)

if __name__=='__main__':unittest.main()
