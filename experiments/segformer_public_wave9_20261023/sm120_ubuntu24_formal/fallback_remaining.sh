#!/usr/bin/env bash
set -u
BASE=/mnt/d/论文复现/experiments/segformer_public_wave9_20261023/sm120_ubuntu24_formal
while [ ! -f "$BASE/results/PSPNet_ResNet50_exit.json" ]; do sleep 60; done
code=$(/root/sm120/venv/bin/python -c "import json; print(json.load(open('$BASE/results/PSPNet_ResNet50_exit.json'))['exit_code'])" 2>/dev/null || echo 99)
if [ "$code" != "0" ] && ! pgrep -f "$BASE/run_queue.sh" >/dev/null; then
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/cu13/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/nccl/lib:/root/sm120/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib
  export PYTHONPATH=/mnt/d/论文复现/external/PaddleSeg-2.10.0
  PY=/root/sm120/venv/bin/python
  TRAIN=/mnt/d/论文复现/external/PaddleSeg-2.10.0/tools/train.py
  CFG=$BASE/configs
  for pair in "DeepLabV3P_ResNet50 linux_deeplabv3p_r50_crack500.yml" "FastSCNN linux_fastscnn_crack500.yml"; do
    set -- $pair; name=$1; cfg=$2; out=$BASE/$name; log=$BASE/logs/"$name"_fallback_train.log
    start=$(date -Iseconds)
    "$PY" "$TRAIN" --config "$CFG/$cfg" --device gpu --iters 18000 --batch_size 4 --do_eval --save_interval 1000 --log_iters 100 --num_workers 0 --seed 42 --save_dir "$out" >>"$log" 2>&1
    rc=$?; end=$(date -Iseconds)
    printf '{"model":"%s","start":"%s","end":"%s","exit_code":%d,"iters":18000,"device":"gpu","seed":42,"fallback":true}\n' "$name" "$start" "$end" "$rc" > "$BASE/results/"$name"_exit.json"
  done
fi