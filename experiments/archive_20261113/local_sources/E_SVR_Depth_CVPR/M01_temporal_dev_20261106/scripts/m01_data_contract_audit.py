#!/usr/bin/env python3
"""Inspect real M01 HDF5 sources and decode samples, without asserting contracts."""
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svr_depth.evaluation.m01_v31.hdf5_preflight import _h5_text, parse_split, pose_diagnostics
from svr_depth.datasets.native_tartanair import _h5_blob


def hashed(value):
    return hashlib.sha256(value).hexdigest()


def probe_blob(h5, ref):
    try:
        raw = _h5_blob(h5, ref)
        with Image.open(BytesIO(raw)) as image:
            a = np.array(image)
        return {'status':'decoded', 'node':str(ref), 'sha256':hashed(raw),
                'shape':list(a.shape), 'dtype':str(a.dtype), 'minimum':float(np.min(a)), 'maximum':float(np.max(a))}
    except Exception as exc:
        return {'status':'decode_error','node':str(ref),'error':f'{type(exc).__name__}: {exc}'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--sample-sequences',type=int,default=3)
    p.add_argument('--hash-hdf5',action='store_true',help='Read and hash entire HDF5 contents')
    a=p.parse_args()
    if a.sample_sequences<1:p.error('sample-sequences must be positive')
    config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    out=a.out_dir.expanduser().resolve();out.mkdir(parents=True,exist_ok=False)
    grouped=defaultdict(list)
    for spec in config['sequences']: grouped[spec['dataset']].append(spec)
    if set(grouped)!={'Bonn','TUM','ScanNet','Sintel'}:raise ValueError('Expected all four datasets')
    manifest=[];summaries=[]
    for dataset,specs in sorted(grouped.items()):
        path=Path(specs[0]['hdf5']).expanduser().resolve(strict=True)
        key=specs[0].get('sequences_key','sequences.json')
        split_key=specs[0]['split_key']
        if any(Path(s['hdf5']).expanduser().resolve()!=path or s.get('sequences_key','sequences.json')!=key or s['split_key']!=split_key for s in specs):
            raise ValueError(f'{dataset}: differing source/split inside one group')
        st=path.stat()
        with h5py.File(path,'r') as h5:
            sequences=json.loads(_h5_text(h5,key)); split=dict(parse_split(_h5_text(h5,split_key)))
            fields=Counter();lengths=defaultdict(Counter); camera=Counter(); poses=Counter(); samples=[]
            for spec in specs:
                name=spec['sequence']
                if name not in sequences or name not in split:raise ValueError(f'{dataset}: sequence absent from split or metadata: {name}')
                seq=sequences[name]; images=seq.get('image',[])
                if len(images)!=split[name]:raise ValueError(f'{dataset}: split length mismatch: {name}')
                for k,v in seq.items():
                    fields[k]+=1
                    if isinstance(v,list):lengths[k][str(len(v)-len(images))]+=1
                models=seq.get('camera_model',[])
                if isinstance(models,list) and models:camera[str(models[0])]+=1
                else:camera['absent_or_unrecognized']+=1
                ps=seq.get('cam2w')
                if isinstance(ps,list) and ps:
                    report=pose_diagnostics(ps[0]);poses[report['status']]+=1
                    if report.get('identity'):poses['first_pose_identity']+=1
                else:poses['missing_or_not_list']+=1
                selection=spec.get('frame_selection','full_sequence')
                start=max(0,len(images)-32) if selection=='release_last32' else 0
                ids=images[start:]
                ids_sha=hashed(json.dumps(ids,ensure_ascii=False,separators=(',',':')).encode('utf-8'))
                manifest.append({'dataset':dataset,'sequence':name,'scene':spec['scene'],
                                 'frame_selection':selection,'source_frame_count':len(images),
                                 'selected_start':start,'source_frame_ids':ids,'source_frame_ids_sha256':ids_sha,
                                 'data_role':'unverified','development_overlap':dataset=='Sintel' and name.split('/')[-1]=='shaman_3'})
                if len(samples)<a.sample_sequences:
                    probes={}
                    for field in ('image','depth','flow_fwd','flow_bwd','flow_fwd_mask','flow_bwd_mask','flow','flow_mask'):
                        arr=seq.get(field)
                        if isinstance(arr,list) and arr:
                            idx=min(start,len(arr)-1)
                            probes[field]=probe_blob(h5,arr[idx]) if arr[idx] is not None else {'status':'null'}
                    sample={'sequence':name,'frame_index':start,'decoded':probes,
                            'camera_model':models[start] if isinstance(models,list) and start<len(models) else None,
                            'camera_params':seq.get('camera_params',[None])[start] if isinstance(seq.get('camera_params'),list) and start<len(seq['camera_params']) else None,
                            'K':seq.get('K',[None])[start] if isinstance(seq.get('K'),list) and start<len(seq['K']) else None,
                            'pose':pose_diagnostics(ps[start] if isinstance(ps,list) and start<len(ps) else None)}
                    samples.append(sample)
        content_sha='not_computed'
        if a.hash_hdf5:
            digest=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
            content_sha=digest.hexdigest()
        report={'dataset':dataset,'hdf5':str(path),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,
                'hdf5_content_hash':content_sha,'sequence_key':key,'split_key':split_key,
                'sequence_count':len(specs),'metadata_fields_count':dict(fields),
                'field_length_minus_rgb_count':{k:dict(v) for k,v in lengths.items()},
                'first_frame_camera_models':dict(camera),'pose_metadata_status':dict(poses),'decoded_samples':samples,
                'status':'metadata_and_sample_decode_only_not_verified',
                'unverified':['flow_direction','flow_index','flow_units','mask_semantics','camera_rectification','pose_provenance','timestamp_sync']}
        (out/(dataset+'_contract.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf8')
        summaries.append({'dataset':dataset,'sequence_count':len(specs),'hdf5':str(path),'hdf5_sha256':content_sha,'camera_models':dict(camera),
                          'field_counts':dict(fields),'pose_status':dict(poses)})
        print(dataset, len(specs), dict(fields), dict(poses), flush=True)
    (out/'manifest.json').write_text(json.dumps({'scope':'candidate_not_frozen','config_sha256':hashed(a.config.read_bytes()),
        'sequences':manifest},indent=2,ensure_ascii=False),encoding='utf8')
    (out/'contract_summary.json').write_text(json.dumps({'datasets':summaries,'formal_contract_verified':False},indent=2,ensure_ascii=False),encoding='utf8')
    print('AUDIT',out,flush=True)


if __name__=='__main__': main()
