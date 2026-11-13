import csv,json,tempfile,unittest
from pathlib import Path
from contract import digest,validate_config
from report import report
import test_m01_contract as m01_fixture

def jwrite(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj),encoding='utf-8')

class ReportTest(unittest.TestCase):
    def test_complete_quality_timing_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=m01_fixture.M01Tests().config();plan['blindness']='HISTORY_UNVERIFIED_NOT_BLIND_VERIFIED';plan['units']=validate_config(plan);plan['sequences'][0]['RGB']=3
            pp=root/'plan.json';jwrite(pp,plan);quality=root/'quality_run';complete_hashes={}
            for unit in plan['units']:
                uid=unit['id'];d=quality/'quality/new_scene'/uid;depth=d/'depth';depth.mkdir(parents=True)
                rows=[];files={}
                for i in range(3):
                    rel=f'quality/new_scene/{uid}/depth/{i:06d}.npz';p=quality/rel;p.write_bytes(b'prediction');files[rel]=digest(p)
                    matched=i<2;value='' if i==1 and unit['family']=='P' else ('0.1' if matched else '')
                    rows.append(dict(index=i,gt_status='MATCHED' if matched else 'UNMATCHED',mask_sha256='m' if matched else '',valid_pixels=1 if matched else 0,AbsRel=value,RMSE='0.2' if value else '',delta1='0.9' if value else ''))
                with (d/'frames.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                jwrite(d/'complete.json',dict(frames_sha256=digest(d/'frames.csv'),prediction_sha256=files));complete_hashes['new_scene/'+uid]=digest(d/'complete.json')
            jwrite(quality/'m04_native_replay.json',[])
            jwrite(quality/'summary.json',dict(plan_sha256=digest(pp),status='QUALITY_ONLY_COMPLETE',completion_hashes=complete_hashes,m04_native_replay_sha256=digest(quality/'m04_native_replay.json')))
            timing=root/'time';th={}
            for unit in plan['units']:
                uid=unit['id']
                for rep in (1,2,3):
                    p=timing/'new_scene'/uid/f'r{rep}.json';p.parent.mkdir(parents=True,exist_ok=True);frames=p.with_name(f'r{rep}.frames.csv');frames.write_text('fixture')
                    jwrite(p,dict(endpoint_clean=True,quality_probes=2,frames=3,mean_ms=10+rep,p50_ms=9,p95_ms=15,peak_allocated_bytes=123,frames_sha256=digest(frames)));th[f'new_scene/{uid}/r{rep}']=digest(p)
            jwrite(timing/'summary.json',dict(status='TIMING_COMPLETE',plan_sha256=digest(pp),quality_summary_sha256=digest(quality/'summary.json'),completion_hashes=th))
            report(pp,quality,root/'report',timing)
            with (root/'report/per_recording.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),10);self.assertTrue(any(x['family']=='P' and x['paired_failed']=='1' for x in rows))
            self.assertTrue(all(float(x['latency_ms'])==12 for x in rows))
            report_summary=json.loads((root/'report/summary.json').read_text());self.assertEqual(report_summary['blindness'],'HISTORY_UNVERIFIED_NOT_BLIND_VERIFIED');self.assertFalse(report_summary['all_scene_groups_verified'])
            with (root/'report/direct_P_comparisons.csv').open() as f:direct=list(csv.DictReader(f))
            self.assertEqual(len(direct),6);self.assertTrue(all(row['P_seed']==row['reference_seed'] for row in direct))
            failed_unit=plan['units'][-1]['id'];meta=quality/'quality/new_scene'/failed_unit/'complete.json';saved=meta.read_bytes();meta.unlink()
            original=json.loads((quality/'summary.json').read_text());incomplete=dict(original,status='QUALITY_INCOMPLETE',completion_hashes={k:v for k,v in complete_hashes.items() if k!='new_scene/'+failed_unit},failed_units=[dict(sequence='new_scene',unit=failed_unit,error='fixture')])
            jwrite(quality/'summary.json',incomplete);report(pp,quality,root/'partial')
            partial=json.loads((root/'partial/summary.json').read_text());self.assertEqual(partial['status'],'PARTIAL_FAILURE_REPORT');self.assertEqual(partial['missing_units'],['new_scene/'+failed_unit]);self.assertFalse((root/'partial/scene_macro.csv').exists())
            meta.write_bytes(saved);jwrite(quality/'summary.json',original)
            p=next(quality.rglob('000000.npz'));p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'prediction changed'):report(pp,quality,root/'bad',timing)

if __name__=='__main__':unittest.main()
