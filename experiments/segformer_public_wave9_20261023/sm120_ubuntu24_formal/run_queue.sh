#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/cu13/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/nccl/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib
export PYTHONPATH=/mnt/d/论文复现/external/PaddleSeg-2.10.0
PY=/root/sm120/venv/bin/python
TRAIN=/mnt/d/论文复现/external/PaddleSeg-2.10.0/tools/train.py
BASE=/mnt/d/论文复现/experiments/segformer_public_wave9_20261023/sm120_ubuntu24_formal
CFG=/mnt/d/论文复现/experiments/segformer_public_wave9_20261023/sm120_ubuntu24_formal/configs
mkdir -p "$BASE/logs" "$BASE/results"
run_one() {
  local name="$1"; local cfg="$2"
  local out="$BASE/$name"
  local log="$BASE/logs/"$name"_train.log"
  local start end code
  start=$(date -Iseconds)
  echo "[QUEUE] START $name $start" | tee -a "$log"
  "$PY" "$TRAIN" --config "$CFG/$cfg" --device gpu --iters 18000 --batch_size 4 --do_eval --save_interval 1000 --log_iters 100 --num_workers 0 --seed 42 --save_dir "$out" >>"$log" 2>&1
  code=$?
  end=$(date -Iseconds)
  printf '{"model":"%s","start":"%s","end":"%s","exit_code":%d,"iters":18000,"device":"gpu","seed":42}\n' "$name" "$start" "$end" "$code" > "$BASE/results/"$name"_exit.json"
  echo "[QUEUE] END $name exit=$code $end" | tee -a "$log"
  return $code
}
run_one PSPNet_ResNet50 linux_pspnet_r50_crack500.yml || exit $?
run_one DeepLabV3P_ResNet50 linux_deeplabv3p_r50_crack500.yml || exit $?
run_one FastSCNN linux_fastscnn_crack500.yml || exit $?
date -Iseconds > "$BASE/results/queue_done.txt"