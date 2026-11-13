#!/usr/bin/env python3
"""Run real GT-depth oracle diagnostics with estimated DIS flow, never GT flow."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from svr_depth.datasets.benchmark import iter_sequence
from svr_depth.evaluation.m01_metrics import array_digest
from svr_depth.evaluation.m01_v31.flow_provider import dis_pair
from svr_depth.evaluation.m01_v31.gt_oracle import evaluate_gt_oracle
from svr_depth.evaluation.m01_v31.hdf5_preflight import pose_diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--fb-threshold-px',type=float,required=True,help='Development-only explicit FB consistency threshold')
    p.add_argument('--pixel-center',choices=('integer','half'),required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--negative-controls',action='store_true',help='Also score deliberately reversed flow direction')
    p.add_argument('--max-attempts-per-dataset',type=int,default=10)
    a=p.parse_args()
    if a.max_attempts_per_dataset<1:p.error('max attempts must be positive')
    config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    out=a.out_dir.expanduser().resolve();out.mkdir(parents=True,exist_ok=False)
    grouped={}
    for spec in config['sequences']:grouped.setdefault(spec['dataset'],[]).append(spec)
    if set(grouped)!={'Bonn','TUM','ScanNet','Sintel'}:raise ValueError('Expected four datasets')
    results=[]
    for dataset,specs in sorted(grouped.items()):
        attempts=[];success=None
        for spec in specs[:a.max_attempts_per_dataset]:
            try:
                it=iter(iter_sequence(spec));prev=next(it);curr=next(it)
                prev_pose=pose_diagnostics(prev.pose);curr_pose=pose_diagnostics(curr.pose)
                if not prev_pose.get('strict_valid') or not curr_pose.get('strict_valid'):
                    attempts.append({'sequence':spec['sequence'],'status':'invalid_pose_matrix','prev_pose':prev_pose,'curr_pose':curr_pose});continue
                if prev.gt is None or curr.gt is None or prev.gt_error or curr.gt_error:
                    attempts.append({'sequence':spec['sequence'],'status':'missing_gt'});continue
                pair=dis_pair(prev.rgb,curr.rgb,threshold_px=a.fb_threshold_px)
                metrics=evaluate_gt_oracle(prev.gt,curr.gt,prev.gt_valid_mask,curr.gt_valid_mask,
                    pair['flow_bwd_px'],pair['flow_bwd_valid'],prev.intrinsics,prev.pose,curr.pose,
                    pixel_center=a.pixel_center,device=a.device)
                if a.negative_controls:
                    wrong=evaluate_gt_oracle(prev.gt,curr.gt,prev.gt_valid_mask,curr.gt_valid_mask,
                        pair['flow_fwd_px'],pair['flow_fwd_valid'],prev.intrinsics,prev.pose,curr.pose,
                        pixel_center=a.pixel_center,device=a.device)
                    metrics['wrong_direction_flow_control']=wrong
                stem=dataset.lower()+'_pair.npz'
                cache=out/stem
                arrays={k:v for k,v in pair.items() if isinstance(v,np.ndarray)}
                with cache.open('xb') as f:np.savez_compressed(f,**arrays)
                success={'dataset':dataset,'sequence':spec['sequence'],'status':metrics['status'],
                    'prev_source_frame_id':prev.source_frame_id,'curr_source_frame_id':curr.source_frame_id,
                    'prev_source_index':prev.source_frame_index,'curr_source_index':curr.source_frame_index,
                    'prev_rgb_sha256':array_digest(prev.rgb),'curr_rgb_sha256':array_digest(curr.rgb),
                    'prev_gt_sha256':array_digest(prev.gt),'curr_gt_sha256':array_digest(curr.gt),
                    'prev_pose_sha256':array_digest(np.asarray(prev.pose)),'curr_pose_sha256':array_digest(np.asarray(curr.pose)),
                    'flow_fwd_sha256':array_digest(pair['flow_fwd_px']), 'flow_bwd_sha256':array_digest(pair['flow_bwd_px']),
                    'flow_fwd_valid_sha256':array_digest(pair['flow_fwd_valid']), 'flow_bwd_valid_sha256':array_digest(pair['flow_bwd_valid']),
                    'camera_sha256':hashlib.sha256(json.dumps(prev.intrinsics,sort_keys=True,default=lambda x:np.asarray(x).tolist()).encode()).hexdigest(),
                    'flow_source':pair['source'],'opencv_version':pair['opencv_version'],
                    'fb_threshold_px':pair['fb_threshold_px'],'pixel_center':a.pixel_center,
                    'camera_model':prev.intrinsics.get('model') if isinstance(prev.intrinsics,dict) else 'K',
                    'camera_metadata':prev.intrinsics,'pose_source':'HDF5 cam2w; provenance unverified',
                    'flow_cache':stem,'flow_cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),
                    'metrics':metrics,'attempts_before_success':attempts,
                    'scope':'diagnostic_estimated_DIS_flow_and_unverified_camera_pose_not_author_oracle'}
                break
            except Exception as exc:
                attempts.append({'sequence':spec['sequence'],'status':'exception','error':f'{type(exc).__name__}: {exc}'})
        if success is None:success={'dataset':dataset,'status':'unavailable','attempts':attempts}
        results.append(success)
        (out/(dataset+'_oracle.json')).write_text(json.dumps(success,indent=2,ensure_ascii=False,default=str,allow_nan=False),encoding='utf-8')
        print(dataset,success['status'],success.get('metrics',{}),flush=True)
    summary={'scope':'development_real_GT_depth_estimated_DIS_not_GT_flow','results':results,
             'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
             'all_datasets_scored':all(r['status']=='ok' for r in results),
             'gt_flow_or_author_oracle_verified':False,'pose_and_camera_provenance_verified':False}
    (out/'oracle_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str,allow_nan=False),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    if not summary['all_datasets_scored']:raise SystemExit(2)


if __name__=='__main__':main()
