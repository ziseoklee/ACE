#!/usr/bin/env bash
set -euo pipefail

ACE_BACKBONES=${ACE_BACKBONES:-"sd15 sd21"}
ACE_SAMPLERS=${ACE_SAMPLERS:-"nr fkc ace"}
ACE_IMAGE_ROOT=${ACE_IMAGE_ROOT:-"output/COCO-MIG_Bench"}
ACE_RESULT_ROOT=${ACE_RESULT_ROOT:-"results/COCO-MIG_Bench"}
ACE_NUM_ITERS=${ACE_NUM_ITERS:-8}

for backbone in ${ACE_BACKBONES}; do
  for sampler in ${ACE_SAMPLERS}; do
    run_name="${backbone}_${sampler}"
    echo "Evaluating ${run_name}"
    uv run python run_migbench_ace.py \
      --image_dir "${ACE_IMAGE_ROOT}/${run_name}" \
      --metric_name "${run_name}" \
      --need_clip_score \
      --need_local_clip \
      --need_success_ratio \
      --need_instance_success_ratio \
      --need_miou_score \
      --output_dir "${ACE_RESULT_ROOT}" \
      --num_iters "${ACE_NUM_ITERS}"
  done
done
