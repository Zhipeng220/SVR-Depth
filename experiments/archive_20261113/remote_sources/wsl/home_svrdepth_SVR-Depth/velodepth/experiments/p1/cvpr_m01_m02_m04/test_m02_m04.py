import csv, json, tempfile, unittest
from pathlib import Path
from m01 import sha, rows
from m02_m04 import freeze, report

def csvout(path, records):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)

class Tests(unittest.TestCase):
    def test_plans_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); rgb=p/'rgb'; rgb.write_text('rgb'); gt=p/'gt'; gt.write_text('gt')
            man=p/'manifest.csv'; csvout(man,[dict(frame_index_zero_based=0,rgb_path=str(rgb),gt_match_status='MATCHED',depth_path=str(gt)),dict(frame_index_zero_based=1,rgb_path=str(rgb),gt_match_status='UNMATCHED',depth_path='')])
            cp={m:[dict(seed=i,path='unused') for i in range(3)] for m in ('O_fast','O_multi','P')}
            parent=dict(protocol='M01-v1',test_unsealed=False,independence_audited=True,depth_contract_confirmed=True,input_sha256={str(x):sha(x) for x in (rgb,gt,man)},checkpoints=cp,sequences=[dict(sequence=s,dataset=s,physical_scene_group='group',manifest=str(man)) for s in ('Bonn','TUM')])
            pp=p/'m01.json'; pp.write_text(json.dumps(parent)); evidence=p/'selection'; evidence.write_text('development records')
            points=[]
            for fam in ('A','O_fast','O_multi','P'):
                for k in (16,32,64):
                    q=dict(id=f'{fam}_{k}',family=fam,k=k,role='primary' if fam=='P' and k==16 else 'selected',parameter_semantics='specified')
                    if fam=='P' and k==16:q.update(gain=1,frequency=1,alignment='raw')
                    points.append(q)
            base=dict(test_unsealed=False,m00_state_and_depth_audit_complete=True,selection_before_test=True)
            configs=[dict(base,experiment='M02',development_access_equal=True,development_selection_evidence=str(evidence),candidate_counts=dict.fromkeys(('A','O_fast','O_multi','P'),3),points=points),dict(base,experiment='M04',same_head_identity_and_state_semantics_audited=True,k=16,gain=1,frequency=1,alignment='raw')]
            for c in configs:
                cfg=p/(c['experiment']+'config'); cfg.write_text(json.dumps(c)); plan=p/(c['experiment']+'plan'); freeze(cfg,pp,plan)
                units=json.loads(plan.read_text())['units']; fr=[]; ts=[]
                for seq in ('Bonn','TUM'):
                    for u in units:
                        ident=dict(sequence=seq,unit=u['id'],seed=u['seed'])
                        fr += [dict(ident,index=0,status='ok',mask_sha256='mask',valid_pixels=3,AbsRel=.1,RMSE=.2,delta1=.9),dict(ident,index=1,status='unmatched',mask_sha256='',valid_pixels='',AbsRel='',RMSE='',delta1='')]
                        ts += [dict(ident,repeat=i,verified='true',mean_ms=11+i) for i in range(3)]
                fp=p/(c['experiment']+'frames'); tp=p/(c['experiment']+'times'); csvout(fp,fr); csvout(tp,ts)
                report(plan,fp,tp,p/(c['experiment']+'out'))
                self.assertEqual(len(rows(p/(c['experiment']+'out')/'per_recording.csv')),2*len(units))
                fr[-2]['mask_sha256']='wrong'; csvout(fp,fr)
                with self.assertRaisesRegex(ValueError,'mask mismatch'): report(plan,fp,tp,p/(c['experiment']+'bad'))
            configs[0]['points'][-3]['gain']=.5
            bad=p/'badconfig'; bad.write_text(json.dumps(configs[0]))
            with self.assertRaisesRegex(ValueError,'P primary'): freeze(bad,pp,p/'badplan')

if __name__=='__main__': unittest.main()
