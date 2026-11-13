cd ~/SVR-Depth/velodepth || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate svrdepth || exit 1
unset PYTHONPATH
BASE=/home/svrdepth/SVR-Depth/velodepth/outputs/y04_v5c_external_20261109_171333
for dataset in TUM ScanNet Sintel; do
 echo "===== $dataset SMOKE ====="
 python experiments/p1/v5c_external_eval.py "$dataset" "$BASE/${dataset}_smoke" 1 16 > "$BASE/${dataset}_smoke.log" 2>&1 || { echo "FAILED $dataset smoke"; exit 1; }
 python experiments/p1/v5c_external_validate.py "$dataset" "$BASE/${dataset}_smoke" || { echo "FAILED $dataset smoke validation"; exit 1; }
 echo "===== $dataset FULL ====="
 python experiments/p1/v5c_external_eval.py "$dataset" "$BASE/$dataset" 0 0 > "$BASE/${dataset}.log" 2>&1 || { echo "FAILED $dataset full"; exit 1; }
 python experiments/p1/v5c_external_validate.py "$dataset" "$BASE/$dataset" || { echo "FAILED $dataset full validation"; exit 1; }
done
echo 'V5C_EXTERNAL_ALL_PASS'
