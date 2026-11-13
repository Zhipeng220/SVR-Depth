#!/usr/bin/env python3
"""Compare real packed-flow samples with the preserved upstream decoder."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from svr_depth.datasets.benchmark import decode_benchmark_flow
from svr_depth.datasets.native_tartanair import _h5_blob
from svr_depth.evaluation.m01_v31.hdf5_preflight import _h5_text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    source=root/'reference/Velodepth-upstream/velodepth/datasets/utils_decode.py'
    spec=importlib.util.spec_from_file_location('m01_official_utils_decode',source)
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    config=json.loads(args.config.read_text(encoding='utf-8-sig'));selected={}
    for item in config['sequences']:selected.setdefault(item['dataset'],item)
    reports=[]
    class Node:
        def __init__(self,data):self.data=data
        def read(self):return np.frombuffer(self.data,dtype=np.uint8).copy()
    class File:
        def __init__(self,data):self.data=data
        def get_node(self,key):return Node(self.data)
    for dataset,item in sorted(selected.items()):
        with h5py.File(item['hdf5'],'r') as h5:
            metadata=json.loads(_h5_text(h5,item.get('sequences_key','sequences.json')))
            seq=metadata[item['sequence']];idx=max(0,len(seq['image'])-32) if item.get('frame_selection')=='release_last32' else 0
            ref=seq['flow_fwd'][idx];raw=_h5_blob(h5,ref)
        canvas={'gt_fields':set(),(0,0):{'gt_fields':set()}}
        output=upstream.decode_flow(canvas,File(raw),ref,0,name='flow_fwd')
        decoded=output[(0,0)]['flow_fwd'][0].permute(1,2,0).numpy()
        own=decode_benchmark_flow(raw,encoding='velodepth_packed12_rgb',geometry='fraction_of_extent')
        h,w=decoded.shape[:2];expected=decoded*np.array([w-1,h-1],np.float32)
        diff=np.abs(expected-own)
        reports.append({'dataset':dataset,'sequence':item['sequence'],'index':idx,'node':ref,
                        'raw_sha256':hashlib.sha256(raw).hexdigest(),'shape':[h,w],
                        'max_abs_difference_px':float(diff.max()),'mean_abs_difference_px':float(diff.mean()),
                        'encoding_assumption':'velodepth_packed12_rgb',
                        'scope':'decode_numerical_parity_does_not_prove_source_units_or_direction'})
    result={'upstream_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'max_allowed_difference_px':1e-4,'results':reports,
            'pass':all(r['max_abs_difference_px']<=1e-4 for r in reports),
            'geometry_contract_verified':False}
    with args.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2));return 0 if result['pass'] else 2


if __name__=='__main__':raise SystemExit(main())
