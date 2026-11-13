#!/usr/bin/env python3
"""Compare raw-grid depth thresholds against upstream AnnotationMask only.

This does not compare upstream resizing, cropping, interpolation, camera
geometry or any other pre/postprocessing; it cannot establish full parity.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from svr_depth.datasets.benchmark import _h5_depth
from svr_depth.datasets.native_tartanair import _h5_blob
from svr_depth.evaluation.m01_v31.hdf5_preflight import _h5_text


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    source=root/'reference/Velodepth-upstream/velodepth/datasets/pipelines/formating.py'
    spec=importlib.util.spec_from_file_location('m01_reference_masker',source)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'));grouped=defaultdict(list)
    for row in cfg['sequences']:grouped[(row['dataset'],row['hdf5'],row.get('sequences_key','sequences.json'))].append(row)
    reports=[]
    for (dataset,path,key),rows in sorted(grouped.items()):
        with h5py.File(path,'r') as h5:
            seqs=json.loads(_h5_text(h5,key))
            for row in rows:
                seq=seqs[row['sequence']];idx=max(0,len(seq['image'])-32) if row.get('frame_selection')=='release_last32' else 0
                depth=_h5_depth(_h5_blob(h5,seq['depth'][idx]),float(row['depth_scale']))
                m01=np.isfinite(depth)&(depth>float(row['min_depth_m']))&(depth<=float(row['max_depth_m']))
                tensor=torch.from_numpy(depth.copy())[None,None]
                obj={'gt_fields':{'depth'},'mask_fields':set(),'sequence_fields':[(0,0)],'depth':{(0,0):tensor}}
                official=ref.AnnotationMask(min_value=0.,max_value=float(row['max_depth_m']),
                    custom_fn=lambda mask,info:mask)(obj)['depth_mask'][(0,0)][0,0].numpy()
                diff=m01^official
                reports.append({'dataset':dataset,'sequence':row['sequence'],'source_index':idx,
                                'm01_pixels':int(m01.sum()),'upstream_threshold_pixels':int(official.sum()),
                                'xor_pixels':int(diff.sum()),'pixels':int(diff.size)})
    result={'scope':'raw_grid_upstream_AnnotationMask_threshold_only_not_full_preprocessing_parity',
            'upstream_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sequences':reports,
            'aggregate':{d:{'sequences':len(group),'sequences_with_difference':sum(r['xor_pixels']>0 for r in group),
                            'xor_pixels':sum(r['xor_pixels'] for r in group)}
                         for d in sorted({r['dataset'] for r in reports}) for group in [[r for r in reports if r['dataset']==d]]},
            'full_mask_parity_verified':False}
    with a.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['aggregate'],indent=2));print('OUTPUT',a.out)


if __name__=='__main__':main()
