#!/usr/bin/env python3
"""Read-only evidence audit for M01/M02/M04; no data/model inference."""
import hashlib,json,os,sys,time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    root=Path.cwd().resolve(); sys.path[:0]=[str(root),str(root/'experiments/p1')]
    import cvpr_m05_formal as m05
    snapshot=Path(os.environ.get('VDEEP_SNAPSHOT','/nonexistent')).resolve()
    result={'experiment':'M01_M02_M04_SOURCE_AUDIT','test_set_accessed':False,'model_inference_performed':False,'training_performed':False,'source_checks':{},'assets':{},'candidate_checkpoints':[],'formal_ready':False}
    for rel,expected in m05.PINNED_SOURCES.items():
        p=root/rel; actual=sha(p) if p.is_file() else None
        result['source_checks'][rel]={'present':p.is_file(),'sha256':actual,'matches_reviewed':actual==expected}
    for rel in ('experiments/p1/y08y09_models.py','experiments/p1/y10_runtime.py','experiments/p1/gt4a2_y11_fallback.py'):
        p=root/rel; result['source_checks'][rel]={'present':p.is_file(),'sha256':sha(p) if p.is_file() else None,'matches_reviewed':None}
    for key,p in [('base_weights',snapshot/'model.safetensors'),('snapshot_config',snapshot/'config.json'),('P_historical',root/m05.KNOWN_P_CHECKPOINT),('O_multi_historical',root/'outputs/gt4a2_y04_validation_fix3_20261025_225902/training/O_multi/Y04_O_multi_final_step1000.pt'),('crowd2_manifest',root/'outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv')]:
        exists=p.is_file(); result['assets'][key]={'path':str(p),'present':exists,'sha256':sha(p) if exists else None}
    outputs=root/'outputs'
    if outputs.exists():
        for directory,dirs,files in os.walk(outputs,followlinks=False):
            dirs[:]=[d for d in dirs if d.lower() not in {'.git','cache','caches','scene_caches','predictions','rgb','depth','wandb'}]
            for name in files:
                p=Path(directory)/name; low=str(p).lower()
                if p.suffix.lower() in ('.pt','.pth','.safetensors') and ('o_fast' in low or 'p_multi' in low or 'o_multi' in low):
                    result['candidate_checkpoints'].append({'path':str(p),'bytes':p.stat().st_size})
    result['candidate_checkpoints'].sort(key=lambda x:x['path'])
    result['limitations']=['Filename candidates are not verified independent training seeds or checkpoint provenance.','Independent test datasets, calibration, access history and M02 development selection are not audited by this scan.','Formal M01/M02/M04 inference is not provided by this script.']
    out=root/'outputs'/('cvpr_main_source_audit_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())); out.mkdir(parents=True,exist_ok=False)
    (out/'source_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('[AUDIT] reviewed sources matching:',sum(x['matches_reviewed'] is True for x in result['source_checks'].values()),'/',len(m05.PINNED_SOURCES))
    for k,v in result['assets'].items(): print('[ASSET]',k,'present='+str(v['present']),v['path'])
    print('[AUDIT] candidate head checkpoints:',len(result['candidate_checkpoints']))
    print('[AUDIT] No formal readiness claimed; see',out/'source_audit.json')
    return 0

if __name__=='__main__': raise SystemExit(main())
