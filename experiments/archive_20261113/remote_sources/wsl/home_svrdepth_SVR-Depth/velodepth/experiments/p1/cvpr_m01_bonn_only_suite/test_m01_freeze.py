import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import prepare
from contract import digest,REQUIRED_DEV_REFERENCES
import test_m01_contract as fixture

class FreezeTest(unittest.TestCase):
 def test_unverified_plan_freeze_and_overlap_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td).resolve();c=fixture.M01Tests().config();c['snapshot']=str(root/'snapshot');snap=Path(c['snapshot']);snap.mkdir();(snap/'model.safetensors').write_bytes(b'model');(snap/'config.json').write_text('{}')
   for key in ('selection_evidence','calibration_evidence','independence_evidence'):
    p=root/(key+'.json');p.write_text('{}');c[key]=str(p)
   for family in ('P','O_fast','O_multi'):
    for cp in c['checkpoints'][family]:
     p=root/(family+'_'+cp['seed']+'.pt');p.write_bytes((family+cp['seed']).encode());rec=root/(family+'_'+cp['seed']+'.json');rec.write_text('{}');cp.update(path=str(p),sha256=digest(p),training_record=str(rec))
   refs=[];reference_img=None
   for seq in sorted(REQUIRED_DEV_REFERENCES):
    im=root/(seq+'.png');im.write_bytes(('ref_'+seq).encode());m=root/(seq+'.csv')
    with m.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['rgb_path']);w.writeheader();w.writerow({'rgb_path':str(im)})
    refs.append({'sequence':seq,'manifest':str(m)});reference_img=im
   c['known_development_manifests']=refs;rgb=root/'candidate.png';rgb.write_bytes(b'candidate');depth=root/'depth.png';depth.write_bytes(b'depth');manifest=root/'candidate.csv';audit=root/'pairing.json';audit.write_text('{}')
   fields=['frame_index_zero_based','rgb_timestamp','rgb_path','depth_timestamp','depth_path','gt_match_status','gt_reused']
   with manifest.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict(frame_index_zero_based=0,rgb_timestamp=1,rgb_path=str(rgb),depth_timestamp=1,depth_path=str(depth),gt_match_status='MATCHED',gt_reused=False))
   c['sequences'][0].update(manifest=str(manifest),pairing_audit=str(audit));cfg=root/'config.json';cfg.write_text(json.dumps(c));out=root/'plan.json'
   with patch.object(Path,'cwd',return_value=root),patch.object(prepare,'PINS',{}),patch.object(prepare,'BASE_SHA',digest(snap/'model.safetensors')),patch.object(prepare,'verify_cache_pairs',return_value={'status':'CACHE_TENSOR_EQUIVALENCE_VERIFIED'}):
    prepare.freeze(cfg,out);p=json.loads(out.read_text());self.assertEqual(p['scope'],'M01_ONLY');self.assertEqual(p['blindness'],'HISTORY_UNVERIFIED_NOT_BLIND_VERIFIED');self.assertEqual(len(p['units']),10);self.assertFalse(p['test_scores_unseen'])
    rgb.write_bytes(reference_img.read_bytes())
    with self.assertRaisesRegex(ValueError,'byte-identical'):prepare.freeze(cfg,root/'rejected.json')
    self.assertFalse((root/'rejected.json').exists())
if __name__=='__main__':unittest.main()
