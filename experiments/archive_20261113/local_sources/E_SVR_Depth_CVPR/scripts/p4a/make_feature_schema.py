import csv
from pathlib import Path
out=Path('docs/p4a/P4A_FEATURE_SCHEMA.csv');out.parent.mkdir(parents=True,exist_ok=True)
rows=[
('update_l2_global','norm','sqrt(sum_l sum(delta_l**2))','scalar; 4 x [1,512,31,41]','S0 and candidate before commit','YES_FROM_SAVED_FEATURES','epsilon not required for absolute norm'),
('update_l2_relative','norm','||delta_all||2/(||S0_all||2+eps)','scalar','S0 and candidate before commit','YES_FROM_SAVED_FEATURES','epsilon preregister before training'),
('update_l2_layer_1','norm','||delta_0||2','scalar','S0 and candidate before commit','YES_FROM_SAVED_FEATURES','level 0 maps to schema layer 1'),
('update_l2_layer_2','norm','||delta_1||2','scalar','S0 and candidate before commit','YES_FROM_SAVED_FEATURES','level 1 maps to schema layer 2'),
('update_l2_layer_3','norm','||delta_2||2','scalar','S0 and candidate before commit','YES_FROM_SAVED_FEATURES','level 2 maps to schema layer 3'),
('update_l2_layer_4','norm','||delta_3||2','scalar','S0 and candidate before commit','YES_FROM_SAVED_FEATURES','level 3 maps to schema layer 4'),
('layerwise_direction_cosine','direction','cos(flat(delta_l),flat(S0_l)) per level','4 scalars','S0 and candidate before commit','COMPUTABLE_NOT_YET_MATERIALIZED','requires nonzero norms; same as alignment proxy below; avoid duplicate columns'),
('normalized_update_direction','direction','delta_l/(||delta_l||2+eps)','four tensors, each [1,512,31,41]','S0 and candidate before commit','NOT_APPROVED_AS_LINEAR_INPUT','high-dimensional and leakage/overfit risk; representation must be frozen before use'),
('cross_layer_direction_statistics','direction','cos(mean_HW(delta_l),mean_HW(delta_k)) for l<k','6 scalars, pooled vectors length 512','S0 and candidate before commit','EXPLORATORY_NOT_VALIDATED','requires common channel coordinate meaning across levels; block until architecture evidence'),
('candidate_state_cosine','compatibility','cos(flat(S0_l+delta_l),flat(S0_l)) per level','4 scalars','S0 and candidate before commit','COMPUTABLE_NOT_YET_MATERIALIZED','undefined on zero norm; current observation is already included in S0'),
('normalized_state_update_difference','compatibility','||delta_l||2/(||S0_l||2+eps) per level','4 scalars','S0 and candidate before commit','COMPUTABLE_NOT_YET_MATERIALIZED','related to norm; avoid claiming independent contribution without ablation'),
('candidate_history_alignment','compatibility','cos(flat(delta_l),flat(warped_parent_l))','4 scalars','candidate and warped parent at current time before commit','BLOCKED_WARPED_PARENT_AVAILABILITY','must prove precommit tensor access and identical alignment from code; cannot substitute raw parent'),
('layerwise_state_compatibility','compatibility','candidate_state_cosine_l and normalized_state_update_difference_l','8 scalars','S0 and candidate before commit','COMPUTABLE_NOT_YET_MATERIALIZED','composite, remove duplicated constituents from linear model'),
('current_confidence','confidence','precommit S0 confidence finite distribution: mean,std,min,max','4 scalars','current S0 state before commit','BLOCKED_SEMANTICS_AUDIT','confirm shape, meaning, precommit availability, masks and absence of future input'),
('dataset_id_sequence_id_frame_id','audit_only','metadata used solely for grouping and integrity','metadata','at capture','GROUP_ONLY','never main predictor inputs')]
with out.open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['feature','category','definition','dimensions','time_and_source','availability','notes']);w.writerows(rows)
print('FEATURES',len(rows))
