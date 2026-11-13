"""Compare V1 forward-map rasterization and inverse-flow mask, with debug plots."""
import sys,glob,json
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,'experiments/p1')
import gt4a2_exp1_train_eval as e
import gt2_fixed_clip_repair_train as g
from saoof_correspondence import correspondence_mask

out=Path('outputs/saoof_00_closeout_20261112'); device=torch.device('cuda')
snap=Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
files=sorted(glob.glob('/home/svrdepth/datasets/bonn/rgbd_bonn_person_tracking2/rgb/*.png'))[:2]
model=g.load_model(snap,'off',device); shape=e.shape_info_for(model,Path(files[0]),device)
def get(i): return e.load_frame({'frame_index_zero_based':i,'rgb_path':files[i],'rgb_timestamp':0},shape,device,need_gt=False)
im0,im1=[get(i)['frame_u8'].permute(1,2,0).numpy() for i in (0,1)]
gray0=cv2.cvtColor(im0,cv2.COLOR_RGB2GRAY); gray1=cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY)
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST); ff=dis.calc(gray0,gray1,None); fb=dis.calc(gray1,gray0,None)
h,w=gray0.shape; yy,xx=np.mgrid[:h,:w].astype(np.float32); qx=xx+ff[...,0]; qy=yy+ff[...,1]
inbounds=np.isfinite(qx)&np.isfinite(qy)&(qx>=0)&(qx<=w-1)&(qy>=0)&(qy<=h-1)
sample=cv2.remap(fb,np.where(inbounds,qx,-1).astype(np.float32),np.where(inbounds,qy,-1).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
err=np.linalg.norm(ff+sample,axis=2); limit=1+.05*(np.linalg.norm(ff,axis=2)+np.linalg.norm(sample,axis=2))
accepted=inbounds & np.isfinite(ff).all(axis=2)&np.isfinite(sample).all(axis=2)&(err<=limit)
# Rasterization audit only: forward-splat accepted continuous q to each neighbor with positive bilinear weight.
count=np.zeros((h,w),np.int32); x0=np.floor(np.where(inbounds,qx,0)).astype(np.int32); y0=np.floor(np.where(inbounds,qy,0)).astype(np.int32)
for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
 x=x0+dx; y=y0+dy; weight=(1-np.abs(qx-x))*(1-np.abs(qy-y)); sel=accepted&(x>=0)&(x<w)&(y>=0)&(y<h)&(weight>1e-7)
 np.add.at(count,(y[sel],x[sel]),1)
forward_mask=count>0
inverse,inv_error=correspondence_mask(ff,fb)
intersection=forward_mask&inverse; union=forward_mask|inverse
mag,angle=cv2.cartToPolar(ff[...,0],ff[...,1],angleInDegrees=True)
hsv=np.zeros((h,w,3),np.uint8); hsv[...,0]=(angle/2).astype(np.uint8); hsv[...,1]=255; hsv[...,2]=np.clip(mag*16,0,255).astype(np.uint8)
cv2.imwrite(str(out/'flow_rgb_current.png'),cv2.cvtColor(im1,cv2.COLOR_RGB2BGR))
cv2.imwrite(str(out/'flow_forward_hsv.png'),cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR))
cv2.imwrite(str(out/'flow_fb_error.png'),cv2.applyColorMap(np.uint8(np.clip(err/3,0,1)*255),cv2.COLORMAP_TURBO))
cv2.imwrite(str(out/'flow_forward_mask.png'),np.uint8(forward_mask)*255)
cv2.imwrite(str(out/'flow_inverse_mask.png'),np.uint8(inverse)*255)
dynamic=mag>=np.percentile(mag[np.isfinite(mag)],75)
data={'frames':files,'coordinate_system':'native 434x574; forward p -> continuous q; backward sampled bilinearly at q','image_shape':[h,w],'flow_valid_source_pixels':int(inbounds.sum()),'out_of_bounds_fraction':float(1-inbounds.mean()),'fb_error_all_inbounds_median':float(np.median(err[inbounds])),'fb_error_all_inbounds_p95':float(np.percentile(err[inbounds],95)),'fb_error_all_inbounds_max':float(np.max(err[inbounds])),'forward_accepted_source_fraction':float(accepted.mean()),'forward_splat_current_fraction':float(forward_mask.mean()),'forward_splat_overlapping_footprints_fraction':float((count>1).mean()),'forward_splat_hole_fraction':float((count==0).mean()),'inverse_sampling_current_fraction':float(inverse.mean()),'mask_iou':float(intersection.sum()/union.sum()),'mask_disagreement_fraction':float(np.logical_xor(forward_mask,inverse).mean()),'high_flow_source_coverage':float(accepted[dynamic].mean()),'high_flow_definition':'source-frame magnitude >= 75th percentile','caveat':'Four-neighbor splat is diagnostic only. Overlapping footprints are not physical collision rate. Formal mask uses inverse sampling with grid_sample align_corners=False, independent of depth validity. No GT mask applied.'}
(out/'flow_audit.json').write_text(json.dumps(data,indent=2),encoding='utf8'); print(json.dumps(data,indent=2))
