import os,sys,json,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','4')
import torch
ROOT=Path('/home/svrdepth/SVR-Depth/velodepth');OUT=ROOT/'outputs/p5norm_eng01_layerwise_smoke_fix';sys.path[:0]=[str(ROOT),str(ROOT/'experiments/p1'),str(OUT)]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1
from eng01_cpu_test import test
def main():
 test();torch.set_num_threads(4);device=torch.device('cpu');snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';model=gt2.load_model(snap,'pre_readout',device);model.eval()
 file=ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940/person_tracking2_pairs_v4_candidate.csv';rows=exp1.read_manifest(file,training=False);shape=exp1.shape_info_for(model,Path(rows[0]['rgb_path']),device);frame=exp1.load_frame(rows[1],shape,device,need_gt=False)
 parent=ROOT/'outputs/p5norm_v2_4_independent_replication/runs/Bonn/person_tracking2/hold/anchor_1_tensors.pt';state=torch.load(parent,map_location='cpu',weights_only=True);original={k:v.clone() for k,v in state.items()};checks=[]
 for i in range(1,5):
  a,b=state['s0_'+str(i)],state['s1_'+str(i)];checks.append({'layer':i,'shape':list(a.shape),'dtype':str(a.dtype),'device':str(a.device),'s1_shape':list(b.shape),'s1_dtype':str(b.dtype),'s1_device':str(b.device),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())});assert a.shape==b.shape and a.dtype==b.dtype and len(a.shape)==4 and checks[-1]['finite']
 assert tuple(shape['original_shape'])==(480,640) and tuple(shape['network_shape'])==(434,574)
 with torch.inference_mode():
  features,tokens=model.pixel_encoder(frame['img']);stacked=[model.stacking_fn(features[i:j]).contiguous() for i,j in model.slices_encoder_range];inputs={'image':frame['img'],'features':stacked};expected_original=tuple(inputs['image'].shape[-2:]);expected_common=tuple(inputs['features'][0].shape[1:3]);model.pixel_decoder.set_shapes(inputs,[{'T':1,'B':1}])
 dec=model.pixel_decoder;assert tuple(dec.radial_module.original_shapes)==expected_original==tuple(shape['network_shape']);assert tuple(dec.common_shape)==expected_common and tuple(dec.radial_module.shapes)==expected_common and tuple(dec.angular_module.shapes)==expected_original
 assert all(torch.equal(state[k],v) for k,v in original.items());assert expected_common==tuple(state['s0_1'].shape[-2:])
 result={'G0_1':'PASS','G0_2':'PASS','G0_3':'PASS','G0_4':'PASS','G0_5':'PASS','G0_6':'PASS','network_shape':list(shape['network_shape']),'common_shape':list(expected_common),'decoder_original_shapes':list(dec.radial_module.original_shapes),'feature_layers':checks,'method':'actual CPU encoder + normal Decoder.set_shapes, no decoder forward','parent_unchanged':True}
 (OUT/'cpu_g0.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
try:main()
except Exception as e:
 (OUT/'cpu_g0.json').write_text(json.dumps({'status':'FAIL','error':traceback.format_exc()},indent=2));raise
