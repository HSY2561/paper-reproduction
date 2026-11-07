#!/usr/bin/env bash
set -eu
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/cu13/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/nccl/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib
export PYTHONPATH=/mnt/d/论文复现/external/PaddleSeg-2.10.0
export PYTHONUNBUFFERED=1
BASE=/mnt/d/论文复现/experiments/segformer_public_wave9_20261023/sm120_ubuntu24_formal
SOURCE=/mnt/d/论文复现/external/PaddleSeg-2.10.0
EVAL=/mnt/d/论文复现/experiments/segformer_public_wave9_20261023/scripts/public_test.py
PY=/root/sm120/venv/bin/python
exec 9>"$BASE/results/evaluation.lock"
flock -n 9 || exit 75
run_logged() {
  local job="$1"
  shift
  local start end rc
  start=$(date -Iseconds)
  printf '[EVAL] START %s %s\n' "$job" "$start"
  set +e
  "$@" >"$BASE/logs/$job.log" 2>&1
  rc=$?
  set -e
  end=$(date -Iseconds)
  printf '{"job":"%s","start":"%s","end":"%s","exit_code":%d}\n' "$job" "$start" "$end" "$rc" >"$BASE/results/"$job"_exit.json"
  printf '[EVAL] END %s exit=%s %s\n' "$job" "$rc" "$end"
  return "$rc"
}
for pair in 'PSPNet_ResNet50 linux_pspnet_r50_crack500.yml' 'DeepLabV3P_ResNet50 linux_deeplabv3p_r50_crack500.yml' 'FastSCNN linux_fastscnn_crack500.yml'; do
  read -r name cfg <<<"$pair"
  checkpoint="$BASE/frozen/$name/model.pdparams"
  test -f "$BASE/frozen/$name/selection.json"
  run_logged "$name"_independent_val "$PY" "$SOURCE/tools/val.py" --config "$BASE/configs/$cfg" --model_path "$checkpoint" --device gpu --num_workers 0
  run_logged "$name"_frozen_test "$PY" "$EVAL" --config "$BASE/configs/test_$name.yml" --checkpoint "$checkpoint" --model "$name" --out "$BASE/results/"$name"_test.json"
done
date -Iseconds >"$BASE/results/evaluation_done.txt"
