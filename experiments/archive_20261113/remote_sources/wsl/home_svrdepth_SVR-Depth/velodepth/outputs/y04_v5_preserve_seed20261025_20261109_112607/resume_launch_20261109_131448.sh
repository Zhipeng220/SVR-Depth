cd ~/SVR-Depth/velodepth || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate svrdepth || exit 1
unset PYTHONPATH;export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20261025 CUDA_VISIBLE_DEVICES=0
RUN="$PWD/outputs/y04_v5_preserve_seed20261025_20261109_112607"
if pgrep -f '^python experiments/p1/y04_v5_trainer_v2.py .*--run-dir .*/y04_v5_preserve_seed20261025_20261109_112607' >/dev/null; then echo 'TRAINING ALREADY RUNNING';exit 1;fi
test -f "$RUN/recovery.pt" || exit 1
LOG="$RUN.resume_$(date +%Y%m%d_%H%M%S).log";echo "RESUME_LOG $LOG"
python experiments/p1/y04_v5_trainer_v2.py --mode adapt --profile bonn13 --manifest-dir "$PWD/outputs/y04_v4_manifest_candidate_20261108_190940" --snapshot "$HOME/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c" --init-head "$PWD/outputs/gt4a2_c3/gt4a2_c3_initial_head.pt" --protocol-file "$PWD/experiments/p1/y04_v4_bonn_indomain_protocol_v1.json" --deterministic --slots-per-seq 8 --steps 1000 --seed 20261025 --save-every 25 --cache-root "$PWD/outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/cache" --run-dir "$RUN" --resume 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]};echo "RESUME_EXIT $rc";exit "$rc"