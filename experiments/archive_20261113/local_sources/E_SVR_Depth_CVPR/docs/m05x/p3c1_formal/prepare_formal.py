import json,hashlib
from pathlib import Path
root=Path.cwd(); src=root/'m05x_work/p3c1_pilot'; dest=root/'m05x_work/p3c1_formal';out=root/'docs/m05x/p3c1_formal'
def replace(s,old,new):
 if s.count(old)!=1:raise RuntimeError('unexpected source count: '+old[:80]+' '+str(s.count(old)))
 return s.replace(old,new)
lock=json.loads((root/'docs/m05x/p3c1_pilot/pilot_protocol.json').read_text(encoding='utf8')); frozen=(out/'anchor_manifest.json').read_bytes();selection=json.loads(frozen)
lock.update(stage='P3-C1 formal',anchor_ids=[a['anchor_id'] for a in selection['formal_anchors']],formal_authorized=True,formal_anchor_manifest_sha256=hashlib.sha256(frozen).hexdigest(),layerwise_match_relative_tolerance=selection['norm_relative_tolerance'],bootstrap=selection['bootstrap'])
p=out/'formal_protocol.json'
if p.exists() and json.loads(p.read_text(encoding='utf8'))!=lock:raise RuntimeError('protocol already exists and differs')
if not p.exists():p.write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
s=(src/'capture_p3c1.py').read_text(encoding='utf8')
s=replace(s,"    branch_v4n=m05.clone_state(branch_O);", "    layer_scales=[]\n    for a,b in zip(delta4,delta5):\n        na,nb=norm([a]),norm([b])\n        if min(na,nb)<=c1lock['norm_minimum'] or not math.isfinite(na+nb):raise RuntimeError('layer norm zero or nonfinite')\n        layer_scales.append(nb/na)\n    branch_v4l=m05.clone_state(branch_O);branch_v4l['features']=[base.float()+scale_l*change for base,change,scale_l in zip(same_input,delta4,layer_scales)]\n    for feature,base,reference in zip(branch_v4l['features'],same_input,delta5):\n        actual_l=norm([feature-base.float()]);expected_l=norm([reference])\n        if abs(actual_l-expected_l)/expected_l>c1lock['layerwise_match_relative_tolerance']:raise RuntimeError('layerwise norm mismatch')\n    put_state(arrays,'committed_V4L',branch_v4l)\n    branch_v4n=m05.clone_state(branch_O);")
s=replace(s,"branch_states={'O':branch_O,'V5':branch_T,'V4M':branch_v4m,'V4N':branch_v4n,'S':branch_S}","branch_states={'O':branch_O,'V5':branch_T,'V4M':branch_v4m,'V4L':branch_v4l,'V4N':branch_v4n,'S':branch_S}")
s=replace(s,"    for branch,init in branch_states.items():\n        m05.set_mode", "    events=[]\n    for branch,init in branch_states.items():\n        events.append('replay_start_'+branch);print(events[-1],flush=True)\n        m05.set_mode")
s=replace(s,"# The GT is first loaded after inference is complete.\nmetrics={}; inputs=[]", "        events.append('replay_end_'+branch);print(events[-1],flush=True)\n# The GT is first loaded after inference is complete.\nif len(events)!=12 or events[-1]!='replay_end_S':raise RuntimeError('all replays must finish before GT')\nevents.append('gt_load_start');print(events[-1],flush=True)\nmetrics={}; inputs=[]")
s=replace(s,"'v4_scale_gpu':scale,", "'v4_scale_gpu':scale,'v4_layer_scales_gpu':layer_scales,'events':events,")
(dest/'capture_formal.py').write_text(s,encoding='utf8')
v=(src/'verify_p3c1.py').read_text(encoding='utf8')
v=replace(v,"BRANCHES=('O','V5','V4M','V4N','S')", "BRANCHES=('O','V5','V4M','V4L','V4N','S')")
v=replace(v,"ROOT/'docs/m05x/p3c1_pilot/pilot_protocol.json'", "ROOT/'docs/m05x/p3c1_formal/formal_protocol.json'")
v=replace(v,"ROOT/'m05x_work/p3c1_pilot/capture_p3c1.py'", "ROOT/'m05x_work/p3c1_formal/capture_formal.py'")
v=replace(v," checks['registered_anchor']=", " checks['formal_manifest_integrity']=sha(ROOT/'docs/m05x/p3c1_formal/anchor_manifest.json')==lock['formal_anchor_manifest_sha256']\n checks['registered_anchor']=")
v=replace(v,"  expected_m=f0+np.float32(meta['v4_scale_gpu'])*(f4-f0);maxerr=max(maxerr,float(np.max(np.abs(fm-expected_m))))", "  fl=a[f'committed_V4L__feature{i}'];dl=(fl-f0).astype(np.float64);nl=float(np.sqrt(np.sum(dl*dl)));n5l=math.sqrt(v5);n4l=math.sqrt(v4)\n  if n4l<=lock['norm_minimum'] or n5l<=lock['norm_minimum']:raise ValueError('zero or nonfinite layer norm')\n  layer_scale=n5l/n4l;scale_saved=meta['v4_layer_scales_gpu'][i]\n  layers[-1].update(v4_layerwise_l2=nl,layerwise_relative_error=abs(nl-n5l)/n5l,layerwise_scale=layer_scale,layerwise_formula_max_abs_error=float(np.max(np.abs(fl-(f0+np.float32(scale_saved)*(f4-f0))))),layerwise_scale_match=math.isclose(layer_scale,scale_saved,rel_tol=1e-9,abs_tol=1e-12))\n  expected_m=f0+np.float32(meta['v4_scale_gpu'])*(f4-f0);maxerr=max(maxerr,float(np.max(np.abs(fm-expected_m))))")
v=replace(v," checks['norm_match']=relative<=", " checks['layerwise_norms_and_formulas']=all(x['layerwise_relative_error']<=lock['layerwise_match_relative_tolerance'] and x['layerwise_formula_max_abs_error']<=lock['formula_abs_tolerance'] and x['layerwise_scale_match'] for x in layers)\n checks['event_order']=meta['events']==[e for b in BRANCHES for e in ('replay_start_'+b,'replay_end_'+b)]+['gt_load_start']\n checks['norm_match']=relative<=")
v=replace(v," for h in (1,4):metrics[f'direction_delta_H{h}']=metrics[f'V4M_H{h}']-metrics[f'V5_H{h}']", " for h in (1,4):\n  metrics[f'direction_delta_H{h}']=metrics[f'V4M_H{h}']-metrics[f'V5_H{h}']\n  metrics[f'layerwise_delta_H{h}']=metrics[f'V4L_H{h}']-metrics[f'V5_H{h}']")
(dest/'verify_formal.py').write_text(v,encoding='utf8')
run=(src/'run_p3c1.py').read_text(encoding='utf8')
for old,new in [('m05x_work/p3c1_pilot','m05x_work/p3c1_formal'),('docs/m05x/p3c1_pilot','docs/m05x/p3c1_formal'),("report/'pilot_protocol.json'","report/'formal_protocol.json'"),("scripts/'capture_p3c1.py','capture_p3c1.py'","scripts/'capture_formal.py','capture_formal.py'"),("report/'formal_protocol.json','pilot_protocol.json'","report/'formal_protocol.json','formal_protocol.json'"),("remote+'/capture_p3c1.py'","remote+'/capture_formal.py'"),("remote+'/pilot_protocol.json'","remote+'/formal_protocol.json'"),("scripts/'verify_p3c1.py'","scripts/'verify_formal.py'"),('outputs/m05x_p3c1_','outputs/m05x_p3c1_formal_'),('P3C1_FINISHED','P3C1_FORMAL_FINISHED')]:
 if old not in run:raise RuntimeError('missing runner source: '+old)
 run=run.replace(old,new)
(dest/'run_formal.py').write_text(run,encoding='utf8')
print('prepared, protocol SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
