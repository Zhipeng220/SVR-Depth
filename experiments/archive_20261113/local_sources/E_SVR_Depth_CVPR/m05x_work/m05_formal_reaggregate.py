"""Offline reaggregation of the frozen historical M05 output; never runs a model."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

KEY = ('seed', 'sequence', 'cycle', 'age', 'parent')
PARENTS = ('A_native', 'P_maintained')

def read_csv(path, required):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or set(required) - set(reader.fieldnames):
            raise ValueError(f'{path}: missing columns {set(required) - set(reader.fieldnames or [])}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    return rows

def indexed(rows, fields):
    out = {}
    for row in rows:
        key = tuple(row[x] for x in fields)
        if any(not x for x in key) or key in out:
            raise ValueError(f'empty or duplicate key: {key}')
        out[key] = row
    return out

def number(value):
    try: n = float(value)
    except (ValueError, TypeError): raise ValueError(f'non-numeric value: {value!r}')
    if not math.isfinite(n): raise ValueError(f'non-finite value: {value!r}')
    return n

def aggregate(rows, sequences, ages):
    """Each cycle equal within age, each age equal within sequence, sequences equal."""
    grouped = defaultdict(list)
    for r in rows:
        if r['parent'] not in PARENTS: raise ValueError('unknown parent')
        grouped[(r['seed'], r['parent'], r['sequence'], int(r['age']))].append(number(r['effect']))
    strata = [{'seed': k[0], 'parent': k[1], 'sequence': k[2], 'age': k[3],
               'n': len(v), 'mean': statistics.mean(v)} for k, v in sorted(grouped.items())]
    result=[]
    for seed in sorted({r['seed'] for r in rows}):
        for parent in PARENTS:
            seq_values = {}
            for seq in sequences:
                age_values = []
                for age in ages:
                    values = grouped.get((seed, parent, seq, age), [])
                    if not values: raise ValueError(f'missing stratum: {seed} {parent} {seq} {age}')
                    age_values.append(statistics.mean(values))
                seq_values[seq] = statistics.mean(age_values)
            result.append({'seed': seed, 'parent': parent, 'sequence_means': seq_values,
                           'mean': statistics.mean(seq_values.values())})
    return result, strata

def reaggregate(source, out, descriptive_only=False):
    source, out = source.resolve(), out.resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('output must not overlap input directory')
    if out.exists(): raise FileExistsError(out)
    protocol = json.loads((source/'protocol_lock.json').read_text(encoding='utf-8-sig'))
    anchors = read_csv(source/'m05_anchor_effects.csv', KEY+('H4_T_minus_O_AbsRel',))
    future = read_csv(source/'m05_future_frame_metrics.csv', KEY+('horizon','branch','frame_index','anchor_index','gt_matched','valid_pixels','total_pixels','gt_valid_ratio','AbsRel'))
    current = read_csv(source/'m05_current_frame_metrics.csv', KEY+('branch','anchor_index','current_output_source','AbsRel','RMSE','delta1','valid_pixels','total_pixels','gt_matched'))
    manifest = read_csv(source/'anchor_manifest.csv', ('sequence','cycle','age','future_horizon','future_index','gt_matched'))
    ai=indexed(anchors,KEY); fi=indexed(future,KEY+('horizon','branch')); ci=indexed(current,KEY+('branch',)); mi=indexed(manifest,('sequence','cycle','age','future_horizon'))
    sequences=list(protocol['sequences']); ages=protocol['ages']; seeds=sorted({r['seed'] for r in anchors})
    expected={(seed,seq,str(cycle),str(age),parent) for seed in seeds for seq in sequences for cycle in protocol['selection'][seq]['selected_cycles'] for age in ages for parent in PARENTS}
    if set(ai)!=expected: raise ValueError(f'anchor set mismatch: missing={len(expected-set(ai))} extra={len(set(ai)-expected)}')
    paired=[]; max_diff=0.; max_current_metric_diff=0.
    for key, anchor in ai.items():
        try:
            o,t=(fi[key+('4',branch)] for branch in ('O','T'))
            co,ct=(ci[key+(branch,)] for branch in ('O','T'))
            m=mi[(key[1],key[2],key[3],'4')]
        except KeyError as e: raise ValueError(f'missing O/T pair or target manifest: {e}')
        if any(r['gt_matched'].lower()!='true' for r in (o,t,co,ct,m)): raise ValueError(f'missing GT: {key}')
        if int(o['valid_pixels'])<=0 or int(t['valid_pixels'])<=0: raise ValueError(f'no valid pixels: {key}')
        if o['frame_index']!=t['frame_index'] or o['frame_index']!=m['future_index']: raise ValueError(f'future frame mismatch: {key}')
        if o['anchor_index']!=t['anchor_index'] or int(o['frame_index'])!=int(o['anchor_index'])+4: raise ValueError(f'horizon mismatch: {key}')
        for field in ('valid_pixels','total_pixels','gt_valid_ratio'):
            if number(o[field])!=number(t[field]): raise ValueError(f'GT validity metadata mismatch: {key} {field}')
        if co['current_output_source']!='D1' or ct['current_output_source']!='D1': raise ValueError(f'current output source mismatch: {key}')
        for field in ('AbsRel','RMSE','delta1','valid_pixels','total_pixels'):
            diff=abs(number(co[field])-number(ct[field])); max_current_metric_diff=max(max_current_metric_diff,diff)
            if diff>1e-10: raise ValueError(f'current output metric mismatch: {key} {field}')
        value=number(t['AbsRel'])-number(o['AbsRel']); diff=abs(value-number(anchor['H4_T_minus_O_AbsRel']))
        max_diff=max(max_diff,diff)
        if diff>1e-12: raise ValueError(f'anchor effect mismatch: {key} {diff}')
        paired.append(dict(zip(KEY,key),effect=value))
    per_seed,strata=aggregate(paired,sequences,ages)
    sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    original=json.loads((source/'m05_formal_summary.json').read_text(encoding='utf-8-sig'))
    validation={'paired_anchor_rows':len(paired),'parents':{p:sum(x['parent']==p for x in paired) for p in PARENTS},'max_anchor_delta_error':max_diff,'max_current_metrics_difference':max_current_metric_diff,'h4_target_gt_and_validity_metadata':'PASS','original_recorded_invariants_pass':original.get('all_invariants_pass'),'pixelwise_gt_mask_identity':'UNVERIFIABLE: original CSV has no GT mask hashes or masks','current_D1_tensor_identity':'UNVERIFIABLE from CSV metrics alone; historical run reports invariant pass','strict_invariants_status':'BLOCKED: no pixelwise masks or current D1 tensors/hash available','interpretation':'descriptive only; no inferential confidence intervals'}
    if not descriptive_only:
        raise ValueError('BLOCKED: pixelwise GT mask and current D1 tensor invariants cannot be verified from these CSVs; use --descriptive-only to report the paired descriptive reconstruction')
    out.mkdir(parents=True,exist_ok=False)
    for name,obj in [('corrected_primary_summary.json',per_seed),('validation_report.json',validation),('source_sha256.json',sha)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,rows in [('paired_effects.csv',paired),('strata_summary.csv',strata)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return per_seed,validation

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--descriptive-only',action='store_true');args=parser.parse_args()
    print(json.dumps(reaggregate(args.input_dir,args.output_dir,args.descriptive_only),ensure_ascii=False,indent=2))
