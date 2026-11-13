import csv,hashlib,json,itertools,collections
from pathlib import Path
root=Path.cwd();out=root/'docs/data_audit/provenance_21_20261110_114734';inventory=json.loads((out/'remote_inventory.json').read_text(encoding='utf8'));selection=json.loads((out/'selected_sequences.json').read_text(encoding='utf8'));seqs=selection['sequences'];frames=inventory['frames'];seqinfo={(x['dataset'],x['sequence']):x for x in inventory['sequences']}
def key(x):return x['dataset'],x['sequence']
def savecsv(name,fieldnames,rows):
 with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
ff=['dataset','sequence','index','modality','path','realpath','is_symlink','timestamp','hdf5_key','sha256','decoded_pixel_sha256','phash64','bytes','image_shape','dtype']
savecsv('frame_inventory.csv',ff,[{k:json.dumps(x[k],ensure_ascii=False) if k=='image_shape' else x.get(k,'') for k in ff} for x in frames])
groups=collections.defaultdict(list)
for f in frames:groups[(f['modality'],f['sha256'])].append(f)
exact=[]
for (mod,h),vals in groups.items():
 for a,b in itertools.combinations(vals,2):
  if key(a)==key(b):continue
  exact.append({'modality':mod,'sha256':h,'dataset_a':a['dataset'],'sequence_a':a['sequence'],'index_a':a['index'],'path_a':a['path'],'dataset_b':b['dataset'],'sequence_b':b['sequence'],'index_b':b['index'],'path_b':b['path'],'bytes':a['bytes'],'interpretation':'identical encoded bytes only; one static/depth frame does not prove same recording'})
exactfields=['modality','sha256','dataset_a','sequence_a','index_a','path_a','dataset_b','sequence_b','index_b','path_b','bytes','interpretation']
savecsv('exact_file_duplicates.csv',exactfields,exact)
metadata_records=[]
for x in inventory['sequences']:
 if x['dataset']=='Bonn':
  for m in x.get('metadata',[])+[x.get('pair_manifest',{})]:
   if m.get('sha256'):metadata_records.append({'dataset':x['dataset'],'sequence':x['sequence'],'path':m['path'],'realpath':m.get('realpath',m['path']),'sha256':m['sha256']})
 else:
  path=x['container']+'::'+x['metadata_key'];metadata_records.append({'dataset':x['dataset'],'sequence':x['sequence'],'path':path,'realpath':x['container_realpath']+'::'+x['metadata_key'],'sha256':x['metadata_sha256']})
mdgroups=collections.defaultdict(list)
for m in metadata_records:mdgroups[m['sha256']].append(m)
mddups=[]
for h,items in mdgroups.items():
 for a,b in itertools.combinations(items,2):
  if key(a)==key(b):continue
  same=a['realpath']==b['realpath'];mddups.append({'sha256':h,'dataset_a':a['dataset'],'sequence_a':a['sequence'],'path_a':a['path'],'dataset_b':b['dataset'],'sequence_b':b['sequence'],'path_b':b['path'],'relationship':'SAME_SHARED_OBJECT' if same else 'DISTINCT_PATHS_SAME_BYTES','note':'shared metadata or matching bytes does not by itself establish same physical recording'})
savecsv('exact_metadata_duplicates.csv',['sha256','dataset_a','sequence_a','path_a','dataset_b','sequence_b','path_b','relationship','note'],mddups)
pixels=collections.defaultdict(list)
for f in frames:pixels[(f['modality'],f['decoded_pixel_sha256'])].append(f)
decoded=[]
for (mod,h),vals in pixels.items():
 for a,b in itertools.combinations(vals,2):
  if key(a)==key(b):continue
  decoded.append({'modality':mod,'decoded_pixel_sha256':h,'dataset_a':a['dataset'],'sequence_a':a['sequence'],'index_a':a['index'],'path_a':a['path'],'dataset_b':b['dataset'],'sequence_b':b['sequence'],'index_b':b['index'],'path_b':b['path'],'same_encoded_sha256':a['sha256']==b['sha256'],'interpretation':'identical decoded pixel array and dtype/shape; not sufficient alone for same recording'})
df=['modality','decoded_pixel_sha256','dataset_a','sequence_a','index_a','path_a','dataset_b','sequence_b','index_b','path_b','same_encoded_sha256','interpretation']
savecsv('decoded_frame_duplicates.csv',df,decoded)
by=collections.defaultdict(list)
for f in frames:by[(f['dataset'],f['sequence'],f['modality'])].append((int(f['phash64'],16),f['index'],f['path']))
candidates=[];temporal=[];threshold=3
for a,b in itertools.combinations(seqs,2):
 ka,kb=key(a),key(b)
 for mod in ['rgb','depth']:
  A=by.get((*ka,mod),[]);B=by.get((*kb,mod),[]);Bv=[(x,i,p) for x,i,p in B];hits=set();examples=[]
  for ha,ia,pa in A:
   for hb,ib,pb in Bv:
    dist=(ha^hb).bit_count()
    if dist<=threshold:
     hits.add((ia,ib))
     if len(examples)<12:examples.append({'index_a':ia,'index_b':ib,'distance':dist,'path_a':pa,'path_b':pb})
  best=0;beststart=None
  for i,j in hits:
   if (i-1,j-1) in hits:continue
   n=1
   while (i+n,j+n) in hits:n+=1
   if n>best:best=n;beststart=(i,j)
  if hits:candidates.append({'dataset_a':ka[0],'sequence_a':ka[1],'dataset_b':kb[0],'sequence_b':kb[1],'modality':mod,'phash_method':'DCT 64 bit 32x32; threshold Hamming<=3','comparisons':len(A)*len(B),'candidate_pairs':len(hits),'longest_consecutive_index_run':best,'run_start':json.dumps(beststart),'examples_first_12':json.dumps(examples,ensure_ascii=False),'interpretation':'perceptual near-match candidates only; static scenes/low-texture depth can collide'})
  if hits:temporal.append({'dataset_a':ka[0],'sequence_a':ka[1],'dataset_b':kb[0],'sequence_b':kb[1],'modality':mod,'timestamp_overlap':'UNKNOWN','pixel_order_check':'perceptual candidate index-run only','candidate_pair_count':len(hits),'longest_consecutive_index_run':best,'run_start':json.dumps(beststart),'assessment':'NEEDS_REVIEW' if best>=3 else 'NOT_ESTABLISHED','notes':'pHash is not a temporal timestamp and cannot establish common recording'})
cf=['dataset_a','sequence_a','dataset_b','sequence_b','modality','phash_method','comparisons','candidate_pairs','longest_consecutive_index_run','run_start','examples_first_12','interpretation']
tf=['dataset_a','sequence_a','dataset_b','sequence_b','modality','timestamp_overlap','pixel_order_check','candidate_pair_count','longest_consecutive_index_run','run_start','assessment','notes']
savecsv('perceptual_candidates.csv',cf,candidates)
# Missing candidate rows are not negative proof. Add all 210 pairs when no match.
pairs={(x['dataset_a'],x['sequence_a'],x['dataset_b'],x['sequence_b'],x['modality']) for x in temporal}
for a,b in itertools.combinations(seqs,2):
 for mod in ['rgb','depth']:
  k=(a['dataset'],a['sequence'],b['dataset'],b['sequence'],mod)
  if k not in pairs:temporal.append(dict(zip(tf,[*k,'UNKNOWN','no pHash candidate within threshold',0,0,'','NOT_ESTABLISHED','no near-match does not establish independence'])))
times={}
for x in seqs:
 t=[float(f['timestamp']) for f in frames if key(f)==key(x) and f['modality']=='rgb' and f.get('timestamp')]
 if t:times[key(x)]=(min(t),max(t))
for r in temporal:
 a=times.get((r['dataset_a'],r['sequence_a']));b=times.get((r['dataset_b'],r['sequence_b']))
 r['recorded_rgb_range_a']=json.dumps(a) if a else '';r['recorded_rgb_range_b']=json.dumps(b) if b else ''
 if a and b and r['dataset_a']==r['dataset_b']=='Bonn':
  r['timestamp_overlap']='OVERLAP_IN_PAIR_CSV_RANGES' if max(a[0],b[0])<=min(a[1],b[1]) else 'NO_OVERLAP_IN_PAIR_CSV_RANGES'
  r['notes']+='; pair CSV timebase provenance is not independently certified'
savecsv('temporal_overlap_report.csv',tf+['recorded_rgb_range_a','recorded_rgb_range_b'],temporal)
relations=[]
for a,b in itertools.combinations(seqs,2):
 ka,kb=key(a),key(b);ta=[r for r in temporal if (r['dataset_a'],r['sequence_a'],r['dataset_b'],r['sequence_b'])==(ka[0],ka[1],kb[0],kb[1])];near=max((int(r['longest_consecutive_index_run']) for r in ta),default=0);scene='UNKNOWN';reason='physical source and capture provenance unavailable';evidence=''
 if ka[0]==kb[0]=='TUM' and all(any(n in q for n in ('fr1_desk_val','fr1_desk2_val','fr1_room_val')) for q in (ka[1],kb[1])):
  scene='PROVISIONAL_SAME_PHYSICAL_SCENE';reason='official canonical TUM desk/desk2/room descriptions share four desks; local HDF5 key labels match canonical names but original-to-HDF5 provenance not yet authenticated';evidence='https://webshare.cvg.cit.tum.de/g/rgbd/dataset/'
 if near>=3:reason+='; pHash consecutive candidate run '+str(near)+' requires human review and does not prove overlap'
 relations.append({'dataset_a':ka[0],'sequence_a':ka[1],'dataset_b':kb[0],'sequence_b':kb[1],'recording_relation':'UNKNOWN','session_relation':'UNKNOWN','scene_relation':scene,'evidence':reason,'source_reference':evidence,'independent_recordings_certified':False})
relations.append({'dataset_a':'Bonn','sequence_a':'crowd3 (training; outside 21)','dataset_b':'Bonn','sequence_b':'removing_nonobstructing_box','recording_relation':'UNKNOWN','session_relation':'UNKNOWN','scene_relation':'PROVISIONAL_SHARED_STATIC_GT_MODEL','evidence':'official dataset provides common static-scene ground truth model; local per-sequence cloud files not found and mapping/use not verified','source_reference':'https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html','independent_recordings_certified':False})
savecsv('physical_relationships.csv',['dataset_a','sequence_a','dataset_b','sequence_b','recording_relation','session_relation','scene_relation','evidence','source_reference','independent_recordings_certified'],relations)
anchors=json.loads((root/'docs/m05x/p2b/anchor_manifest.json').read_text(encoding='utf8'))['anchors'];containers={x['path']:x['sha256'] for x in json.loads((out/'container_hashes.json').read_text(encoding='utf8'))['files']}
mapping=[]
for n,x in enumerate(seqs,1):
 ds,seq=x['dataset'],x['sequence'];info=seqinfo.get((ds,seq),{});fs=[r for r in frames if key(r)==(ds,seq)];rgb=[r for r in fs if r['modality']=='rgb'];depth=[r for r in fs if r['modality']=='depth'];af=[a for a in anchors if key(a)==(ds,seq)];ts=[float(q['timestamp']) for q in rgb if q.get('timestamp')]
 metadata=info.get('metadata',[]);find=lambda suffix:next((m for m in metadata if m['path'].endswith('/'+suffix)),{})
 rgbmeta=find('rgb.txt');depthmeta=find('depth.txt');trajectory=find('groundtruth.txt');sample=(rgb or [{}])[0];dsample=(depth or [{}])[0]
 prefix=(sample.get('hdf5_key','').split('/')[0] if ds!='Bonn' else '')
 mapping.append({'record_id':f'{n:02d}','dataset':ds,'selected_sequence_id':seq,'dataset_native_id':'','native_label_candidate':prefix if prefix else seq,'native_label_status':'PROVISIONAL_PATH_LABEL_ONLY','original_dataset_name':{'Bonn':'Bonn RGB-D Dynamic Dataset','TUM':'TUM RGB-D','ScanNet':'ScanNet'}[ds],'source_video_id':'','acquisition_id':'','capture_session_id':'','physical_scene_id':'','source_rgb_path':info.get('container','')+'::RGB embedded' if ds!='Bonn' else info.get('root','')+'/rgb','source_depth_path':info.get('container','')+'::Depth embedded' if ds!='Bonn' else info.get('root','')+'/depth','timestamp_path':rgbmeta.get('path',''),'depth_timestamp_path':depthmeta.get('path',''),'metadata_path':info.get('container','')+'::'+info.get('metadata_key','') if ds!='Bonn' else info.get('pair_manifest',{}).get('path',''),'trajectory_path':trajectory.get('path',''),'trajectory_sha256':trajectory.get('sha256',''),'pointcloud_path':'','rgb_files_sha256':'SEE_frame_inventory.csv','depth_files_sha256':'SEE_frame_inventory.csv','timestamp_sha256':rgbmeta.get('sha256',''),'depth_timestamp_sha256':depthmeta.get('sha256',''),'metadata_sha256':info.get('metadata_sha256','') if ds!='Bonn' else info.get('pair_manifest',{}).get('sha256',''),'original_files_exist':'selected payloads accessible' if rgb else 'UNKNOWN','is_symlink':info.get('container_is_symlink',info.get('root_is_symlink','UNKNOWN')),'realpath':info.get('container_realpath',info.get('root_realpath','')),'recording_start':min(ts) if ts else '', 'recording_end':max(ts) if ts else '', 'acquisition_date':'','camera_device':'','camera_serial':'','recording_fps':'','image_resolution':json.dumps(sample.get('image_shape','')),'depth_resolution':json.dumps(dsample.get('image_shape','')),'frame_count':info.get('frame_count',''),'temporal_range_source':'pair CSV timestamps; identity/timebase unverified' if ts else 'UNKNOWN','P2A':'YES' if any(a.get('pilot_p2a_exposed') for a in af) else 'NO','P2B':'YES','P3A':'YES','P3B':'YES','V4V5_development':'YES','historical_M05':'UNKNOWN','method_selection':'YES','physical_recording_relation':'UNKNOWN','session_relation':'UNKNOWN','scene_relation':'UNKNOWN','independent_test':'NO_DEVELOPMENT','notes':'no source video identity certified; source_id and physical scene require provenance and temporal evidence'})
for row in mapping:
 info=seqinfo[(row['dataset'],row['selected_sequence_id'])];row['container_sha256']=containers.get(info.get('container',''),'');row['original_files_exist']='UNKNOWN_ORIGINAL_SOURCE; PARTIAL_SELECTED_PAIR_REFERENCES' if any(x.get('dataset')==row['dataset'] and x.get('sequence')==row['selected_sequence_id'] for x in inventory['errors']) else 'UNKNOWN_ORIGINAL_SOURCE; SELECTED_REFERENCES_ACCESSIBLE'
mf=list(mapping[0]);savecsv('physical_provenance_mapping.csv',mf,mapping)
lead=inventory.get('tum_depth_lead',[]);lf=['key','sha256','decoded_pixel_sha256','phash64','bytes','image_shape','dtype'];savecsv('known_tum_depth_lead.csv',lf,[{k:json.dumps(x[k]) if k=='image_shape' else x.get(k,'') for k in lf} for x in lead])
stats={'selected_sequences':len(seqs),'selected_by_dataset':dict(collections.Counter(x['dataset'] for x in seqs)),'anchor_count':selection['anchor_count'],'frames':len(frames),'frames_by_dataset_modality':{str(k):v for k,v in collections.Counter((x['dataset'],x['modality']) for x in frames).items()},'unpaired_or_missing':inventory['errors'],'exact_cross_sequence_pairs':len(exact),'decoded_cross_sequence_pairs':len(decoded),'shared_metadata_pair_references':len(mddups),'distinct_metadata_paths_same_bytes':sum(x['relationship']=='DISTINCT_PATHS_SAME_BYTES' for x in mddups),'perceptual_pair_summaries':len(candidates),'perceptual_longest_run_ge3':sum(x['longest_consecutive_index_run']>=3 for x in candidates),'physical_recording_groups_certified':0,'same_session_certified':0,'physical_scenes_certified':0,'tum_depth_lead_source':'docs/m05x/p3c0/video_hash_audit.json','bonn_crowd3_and_removing_pointcloud_status':'NOT_FOUND_IN_NAMED_ROOTS_OR_READONLY_MAXDEPTH4_SCAN; not proof absent','frame_scope':'all frame references in 5 Bonn pair manifests plus unreferenced image files in those directories, and 16 selected HDF5 sequence metadata; not necessarily all original source video frames','perceptual_scope':'all cross-sequence pHash pairs of accessible selected references; Hamming <=3; candidate only'}
(out/'audit_summary.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps(stats,ensure_ascii=False))

