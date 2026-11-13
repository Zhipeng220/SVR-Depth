#!/usr/bin/env python3
"""Create an incomplete, review-required configuration from known local provenance."""
import argparse,json,os
from pathlib import Path
from contract import digest,read_json,require

DATA_SHA='11f30c44b7b38ccf05174225b870eac04b9ced1fc36241a5dd386eab5c79d0e4'
P_SHA='77a86cdf523f776608b6b591d6c6552002b90398e85d00d3a3fe29afe14a0a24'
O_SHA='558e1850d7b08939bef0db0eb4f7a48fdff8a2fb58417cd4c380ee3c6a6f6948'
DEVELOPMENT=('moving_nonobstructing_box','moving_obstructing_box','placing_nonobstructing_box','removing_nonobstructing_box','person_tracking2')

def create(root,out,snapshot):
    root=Path(root).resolve();out=Path(out);require(not out.exists(),'Refusing to overwrite an existing configuration')
    c=read_json(Path(__file__).parent/'config.example.json');c['snapshot']=str(Path(snapshot).resolve()) if snapshot else ''
    refs=[]
    for seq in ('crowd2','crowd3','balloon2'):
        p=root/'outputs/gt4a2_exp1'/f'gt4a2_exp1_{seq}_pairs_used.csv'
        if p.is_file():refs.append(dict(sequence=seq,manifest=str(p)))
    dp=root/'outputs/y11_confirmation_data/dataset_protocol.json'
    if dp.is_file() and digest(dp)==DATA_SHA:
        d=read_json(dp)
        for seq in DEVELOPMENT:
            p=Path(d.get('manifests',{}).get(seq,{}).get('path',''))
            if p.is_file():refs.append(dict(sequence=seq,manifest=str(p.resolve())))
    c['known_development_manifests']=refs
    old=root/'outputs/gt4a2_y04_validation_fix3_20261025_225902/training'
    for family,cp,summary,expected in [
        ('P',old/'P_multi/gt4a2_c3_C_final_step1000.pt',old/'P_multi/gt4a2_c3_C_training_summary.json',P_SHA),
        ('O_multi',old/'O_multi/Y04_O_multi_final_step1000.pt',old/'O_multi/Y04_O_multi_training_summary.json',O_SHA)]:
        if cp.is_file() and summary.is_file() and digest(cp)==expected:
            c['checkpoints'][family].append(dict(seed='20261025',path=str(cp),sha256=expected,training_record=str(summary)))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(c,f,ensure_ascii=False,indent=2);f.write('\n')
    print('[INCOMPLETE CONFIG]',out,'known development reference manifests:',len(refs),'P candidates:',len(c['checkpoints']['P']),'O_multi candidates:',len(c['checkpoints']['O_multi']),flush=True)
    print('Review and complete all missing data, evidence, seeds and points; no audit flags have been set.',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT',''));a=ap.parse_args();create(Path.cwd(),a.out,a.snapshot)
if __name__=='__main__':main()
