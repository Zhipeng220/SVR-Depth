import csv,json,tempfile,unittest
from pathlib import Path
from contract import digest,validate_config
from report import report
from test_contract import example

def jwrite(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj),encoding='utf-8')

class ReportTest(unittest.TestCase):
    def test_complete_quality_timing_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=example();plan['candidate_counts']=dict.fromkeys(('A','O_fast','O_multi','P'),4);plan['points'].append(dict(id='P_ALT',family='P',k=32,gain=.5,frequency=1,alignment='raw',role='alternative',semantics='fixed alternative'));plan['units']=validate_config(plan);plan['sequences'][0]['RGB']=3
            pp=root/'plan.json';jwrite(pp,plan);quality=root/'quality_run';complete_hashes={}
            for unit in plan['units']:
                uid=unit['id'];d=quality/'quality/new_bonn'/uid;depth=d/'depth';depth.mkdir(parents=True)
                rows=[];files={}
                for i in range(3):
                    rel=f'quality/new_bonn/{uid}/depth/{i:06d}.npz';p=quality/rel;p.write_bytes(b'prediction');files[rel]=digest(p)
                    matched=i<2;value='' if i==1 and unit['family']=='P' else ('0.1' if matched else '')
                    rows.append(dict(index=i,gt_status='MATCHED' if matched else 'UNMATCHED',mask_sha256='m' if matched else '',valid_pixels=1 if matched else 0,AbsRel=value,RMSE='0.2' if value else '',delta1='0.9' if value else ''))
                with (d/'frames.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                jwrite(d/'complete.json',dict(frames_sha256=digest(d/'frames.csv'),prediction_sha256=files));complete_hashes['new_bonn/'+uid]=digest(d/'complete.json')
            jwrite(quality/'m04_native_replay.json',[])
            jwrite(quality/'summary.json',dict(plan_sha256=digest(pp),status='QUALITY_ONLY_COMPLETE',completion_hashes=complete_hashes,m04_native_replay_sha256=digest(quality/'m04_native_replay.json')))
            timing=root/'time';th={}
            for unit in plan['units']:
                uid=unit['id']
                for rep in (1,2,3):
                    p=timing/'new_bonn'/uid/f'r{rep}.json';p.parent.mkdir(parents=True,exist_ok=True);frames=p.with_name(f'r{rep}.frames.csv');frames.write_text('fixture')
                    jwrite(p,dict(endpoint_clean=True,quality_probes=2,frames=3,mean_ms=10+rep,p50_ms=9,p95_ms=15,peak_allocated_bytes=123,frames_sha256=digest(frames)));th[f'new_bonn/{uid}/r{rep}']=digest(p)
            jwrite(timing/'summary.json',dict(status='TIMING_COMPLETE',plan_sha256=digest(pp),quality_summary_sha256=digest(quality/'summary.json'),completion_hashes=th))
            report(pp,quality,root/'report',timing)
            with (root/'report/per_recording.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),45);self.assertTrue(any(x['family']=='P' and x['paired_failed']=='1' for x in rows))
            self.assertTrue(all(float(x['latency_ms'])==12 for x in rows))
            with (root/'report/P_Alt_risk.csv').open() as f:alt=list(csv.DictReader(f))
            self.assertEqual({(x['reference'],x['reference_k']) for x in alt},{('A','16'),('A','32'),('O_fast','16'),('O_fast','32'),('P','16')})
            failed_unit=plan['units'][-1]['id'];meta=quality/'quality/new_bonn'/failed_unit/'complete.json';saved=meta.read_bytes();meta.unlink()
            original=json.loads((quality/'summary.json').read_text());incomplete=dict(original,status='QUALITY_INCOMPLETE',completion_hashes={k:v for k,v in complete_hashes.items() if k!='new_bonn/'+failed_unit},failed_units=[dict(sequence='new_bonn',unit=failed_unit,error='fixture')])
            jwrite(quality/'summary.json',incomplete);report(pp,quality,root/'partial')
            partial=json.loads((root/'partial/summary.json').read_text());self.assertEqual(partial['status'],'PARTIAL_FAILURE_REPORT');self.assertEqual(partial['missing_units'],['new_bonn/'+failed_unit]);self.assertFalse((root/'partial/scene_macro.csv').exists())
            meta.write_bytes(saved);jwrite(quality/'summary.json',original)
            p=next(quality.rglob('000000.npz'));p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'prediction changed'):report(pp,quality,root/'bad',timing)

if __name__=='__main__':unittest.main()
