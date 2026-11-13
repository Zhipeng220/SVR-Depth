import csv, json, tempfile, unittest
from pathlib import Path
from m01 import freeze, aggregate, rows

def csvout(path, fields, data):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)

class PipelineTest(unittest.TestCase):
    def test_freeze_report_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); model=d/'model'; model.write_text('model'); source=d/'source'; source.write_text('source')
            rgb=d/'rgb'; rgb.write_text('rgb'); gt=d/'gt'; gt.write_text('gt')
            manifest=d/'pairs.csv'; csvout(manifest,['frame_index_zero_based','rgb_path','gt_match_status','depth_path'],[dict(frame_index_zero_based=0,rgb_path=rgb,gt_match_status='MATCHED',depth_path=gt),dict(frame_index_zero_based=1,rgb_path=rgb,gt_match_status='UNMATCHED',depth_path='')])
            cps={}
            for m in ('P','O_fast','O_multi'):
                cps[m]=[]
                for seed in range(3):
                    cp=d/f'{m}{seed}'; cp.write_text('weight'); cps[m].append(dict(seed=seed,path=str(cp)))
            seqs=[dict(dataset=ds,sequence=ds,physical_scene_group='scene',camera_setup='camera',sensor_model='sensor',recording_session='session',independence_notes='checked',depth_scale_to_meters=.0002,manifest=str(manifest)) for ds in ('Bonn','TUM')]
            cfg=dict(protocol='M01-v1',test_unsealed=False,independence_audited=True,depth_contract_confirmed=True,k=16,gain=1,frequency=1,alignment='raw',model_files=[str(model)],source_files=[str(source)],checkpoints=cps,sequences=seqs)
            config=d/'config.json'; config.write_text(json.dumps(cfg)); protocol=d/'protocol.json'; freeze(config,protocol)
            frames=[]; timing=[]
            for s in ('Bonn','TUM'):
                for m,seed in [('A','native')]+[(m,str(i)) for m in cps for i in range(3)]:
                    frames += [dict(sequence=s,method=m,seed=seed,index=0,status='ok',mask_sha256='mask',valid_pixels=2,AbsRel=.1 if m=='A' else .12,RMSE=.2,delta1=.9),dict(sequence=s,method=m,seed=seed,index=1,status='unmatched',mask_sha256='',valid_pixels='',AbsRel='',RMSE='',delta1='')]
                    timing += [dict(sequence=s,method=m,seed=seed,repeat=i,verified='true',mean_ms=10+i) for i in range(3)]
            fp=d/'frames.csv'; tp=d/'times.csv'; csvout(fp,list(frames[0]),frames); csvout(tp,list(timing[0]),timing)
            aggregate(protocol,fp,tp,d/'out')
            p=[r for r in rows(d/'out'/'per_recording.csv') if r['method']=='P'][0]
            self.assertEqual(float(p['severe_fraction']),1.0); self.assertEqual(int(p['valid_paired']),1)
            frames[1]['status']='ok'; csvout(fp,list(frames[0]),frames)
            with self.assertRaisesRegex(ValueError,'GT pairing mismatch'): aggregate(protocol,fp,tp,d/'bad')
            model.write_text('modified')
            with self.assertRaisesRegex(ValueError,'Frozen input changed'): aggregate(protocol,fp,tp,d/'bad2')

if __name__=='__main__': unittest.main()
