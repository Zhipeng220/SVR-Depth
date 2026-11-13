"""V1.1 same-source official-native-canvas reference input parity check."""
import os,sys,json,glob
from pathlib import Path
os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'
import numpy as np,torch,cv2
sys.path.insert(0,'experiments/p1')
import gt2_fixed_clip_repair_train as gt2
from velodepth.datasets.bonn import Bonn
from velodepth.datasets import collate_fn
from saoof_reference_preprocess import reference_input,official_frame_from_batch
from saoof_native_stream import SAOOFNativeStream
out=Path('outputs/saoof_01_ref_probe_20261112'); out.mkdir(exist_ok=True,parents=True)
cfg=json.loads(Path('configs/velodepth-eval.json').read_text())['data']; old=np.array
class Legacy(np.ndarray):
 def tostring(self,*a,**kw): return self.tobytes(*a,**kw)
def compat(*a,**kw): return old(*a,**kw).view(Legacy)
try:
 np.array=compat
 ds=Bonn(image_shape=cfg['image_shape'],split_file='val.txt',test_mode=True,crop=cfg['crop'],normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.,num_frames=-1,add_of=True)
finally: np.array=old
ds.resizer.sample=False; ds.resizer.image_shape=[434,574]
i=next(i for i,x in enumerate(ds.dataset) if x['sequence_name'].endswith('/person_tracking2_val'))
ds.get_test_idxs=lambda n,k:([78],None)
batch=collate_fn([ds.get_single_sequence(i)],is_batched=False)
frame=official_frame_from_batch(batch,0,torch.device('cuda')); native=frame['img']; assert tuple(native.shape)==(1,3,434,574)
snapshot=Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
model=gt2.load_model(snapshot,'off',torch.device('cuda')); model.pixel_decoder.camera_gt=False
native_model=gt2.load_model(snapshot,'off',torch.device('cuda')); native_model.pixel_decoder.camera_gt=False
stream=SAOOFNativeStream(native_model,model,{'network_shape':(434,574)})
seen=[]
handle=model.pixel_encoder.register_forward_pre_hook(lambda module,args: seen.append(args[0].detach().clone()))
def infer(x,size):
 inp={'image':x,'image_original':x,'validity_mask':torch.ones((1,1,*size),device=x.device,dtype=torch.bool)}
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
  features,tokens=model.pixel_encoder(x)
  inp['features']=[model.stacking_fn(features[a:b]).contiguous() for a,b in model.slices_encoder_range]
  inp['tokens']=[model.stacking_fn(tokens[a:b]).contiguous() for a,b in model.slices_encoder_range]
  model.pixel_decoder.set_shapes(inp,[{'B':1,'T':1}]); rays,r,_,_=model.pixel_decoder.forward_keyframe(inp,[{'B':1,'T':1}])
 rays=rays.reshape(1,*size,3).permute(0,3,1,2); z=r.float()*rays[:,2:3].float()
 return torch.nn.functional.interpolate(z.log(),size=(434,574),mode='bilinear',align_corners=False,antialias=False).exp()
results=[]
for tag,size in [('low',(224,294)),('mid',(294,392))]:
 # Exercise both real reference entry paths using exactly the same official native tensor.
 seen.clear(); official=reference_input(native,size); a=infer(official,size)
 stream.ref_size=size; b=stream._ref(frame)[None,None]
 assert len(seen)==2
 official_input,streaming=seen
 mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]; std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 rgb=((official*std+mean).clamp(0,1)*255).byte()[0].permute(1,2,0).cpu().numpy()
 cv2.imwrite(str(out/f'reference_input_{tag}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
 results.append({'size':list(size),'input_bitwise_equal':bool(torch.equal(official_input,streaming)),'input_max_abs':float((official_input-streaming).abs().max()),'output_shape':list(a.shape),'output_max_abs':float((a-b).abs().max()),'output_finite_positive':bool((torch.isfinite(a)&(a>0)).all())})
handle.remove()
manifest={'protocol':'SAOOF V1.1 Native-Canvas First','source':'official Bonn HDF5 loader person_tracking2_val index 78','native_size':[434,574],'rgb_order':'RGB','normalization':'ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225]','reference_resize':{'mode':'bilinear','align_corners':False,'antialias':False},'output_resize':{'domain':'log_z','mode':'bilinear','align_corners':False,'antialias':False},'camera_gt_native':False,'camera_gt_reference':False,'note':'Parity tests a single shared native tensor; it does not prove independently generated raw-input native canvases are equal.'}
(out/'reference_preprocess_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
(out/'reference_input_parity.json').write_text(json.dumps({'status':'PASS' if all(x['input_bitwise_equal'] and x['output_finite_positive'] and x['output_max_abs']==0 for x in results) else 'FAIL','results':results},indent=2),encoding='utf8')
print(json.dumps(results,indent=2))
