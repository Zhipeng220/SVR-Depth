import contextlib,io,json,tempfile,unittest
from pathlib import Path
from contract import validate_config
from cache_evidence import verify_cache_pairs,order,order_sha

class M01Tests(unittest.TestCase):
 def config(self):
  seeds=('20261025','20261101','20261102');c={'version':'BONN-M01-ONLY-1','role':'heldout_history_unverified','test_scores_unseen':False,'access_history_status':'unknown','calibration_audited':True,'data_provenance_audited':True,'selection_frozen_before_test':True,'depth_encoding':'Bonn_registered_uint16_z_div5000','depth_min_m':.1,'depth_max_m':10.,'selection_evidence':'s','calibration_evidence':'c','independence_evidence':'i','checkpoints':{m:[{'seed':s} for s in seeds] for m in ('P','O_multi','O_fast')},'sequences':[{'sequence':'new_scene','physical_scene_group':'unknown','physical_scene_verified':False,'recording_session':'unknown','sensor_model':'unknown','camera_setup':'unknown','manifest':'m','pairing_audit':'a','audit_notes':'unknown','pretraining_overlap':'unknown'}]}
  c['points']=[{'id':m,'family':m,'k':16,'gain':0 if m=='A' else 1,'frequency':1,'alignment':'raw','role':'primary' if m=='P' else 'selected','semantics':'test'} for m in ('A','O_fast','O_multi','P')];return c
 def test_scope_and_honest_history(self):
  c=self.config();u=validate_config(c);self.assertEqual(len(u),10);self.assertEqual({x['k'] for x in u},{16});self.assertNotIn('M04',{x['experiment'] for x in u})
  c['role']='heldout';
  with self.assertRaisesRegex(ValueError,'confirmed unseen'):validate_config(c)
  c['test_scores_unseen']=True;c['access_history_status']='verified_unseen';self.assertEqual(len(validate_config(c)),10)
 def test_does_not_allow_false_claim(self):
  c=self.config();c['test_scores_unseen']=True
  with self.assertRaisesRegex(ValueError,'explicitly unknown'):validate_config(c)
  c=self.config();c['points'][0]['k']=32
  with self.assertRaisesRegex(ValueError,'K16'):validate_config(c)
 def test_verified_cache_and_corruption(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);c=self.config();pair_hash={s:'pair_'+s for s in ('crowd2','crowd3','balloon2')};indices={s:[0,255] for s in pair_hash}
   for method in ('P','O_multi'):
    for x in c['checkpoints'][method]:
     seed=x['seed'];base=root/seed/('shared' if seed=='20261025' else method);folder=base/'training'/method;folder.mkdir(parents=True,exist_ok=True);cp=folder/'head.pt';cp.write_bytes(b'checkpoint');x.update(path=str(cp),sha256='hash_'+method+'_'+seed,training_record=str(folder/'summary.json'))
     report={'checkpoint_sha256':x['sha256'],'steps':1000,'sample_order_sha256':order_sha(order(seed)),'training_pair_csv_sha256_by_sequence':pair_hash,'training_rgb_indices_by_sequence':indices}
     if method=='P':report['initial_head_state_sha256']='shared_init'
     else:report.update(initial_adapter_state_sha256='shared_init',optimizer_updates=1000,supervised_propagation_frame_exposures=31000)
     Path(x['training_record']).write_text(json.dumps(report),encoding='utf-8')
     for seq in pair_hash:
      d=base/'training/scene_caches'/seq/'cycle_cache';d.mkdir(parents=True,exist_ok=True);meta={'format':'fmt','pretrained_weights_sha256':'base','pair_sha256':pair_hash[seq],'original_source_hashes':{'source':'sha'},'cycles':{}}
      for i in range(8):
       data=f'{seed}:{seq}:{i}'.encode();f=d/f'cycle_{i:02d}.pt';f.write_bytes(data);import hashlib
       meta['cycles'][str(i)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'supervised':31}
      (d/'cache_manifest.json').write_text(json.dumps(meta),encoding='utf-8')
   with contextlib.redirect_stdout(io.StringIO()):r=verify_cache_pairs(c)
   self.assertEqual(r['status'],'CACHE_TENSOR_EQUIVALENCE_VERIFIED');self.assertEqual(len(r['pairs']),3);self.assertTrue(r['pairs'][0]['historical_shared_path'])
   q=Path(c['checkpoints']['O_multi'][1]['path']).parents[2]/'training/scene_caches/crowd2/cycle_cache/cycle_00.pt';q.write_bytes(b'corrupt')
   with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'Cache bytes differ'):verify_cache_pairs(c)
if __name__=='__main__':unittest.main()
