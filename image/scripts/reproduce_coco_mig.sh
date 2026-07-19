#!/usr/bin/env bash
set -euo pipefail

# Reproduce the six NR/FKC/ACE rows for SD1.5 and SD2.1 in Table E.10.
# Override any variable below from the environment to run a subset.
ACE_BACKBONES=${ACE_BACKBONES:-"sd15 sd21"}
ACE_SAMPLERS=${ACE_SAMPLERS:-"nr fkc ace"}
ACE_SEEDS=${ACE_SEEDS:-"42,37,519,609,123,401,780,0"}
ACE_LEVELS=${ACE_LEVELS:-"0,1,2,3,4"}
ACE_OUTPUT_ROOT=${ACE_OUTPUT_ROOT:-"output/COCO-MIG_Bench"}

for backbone in ${ACE_BACKBONES}; do
  for sampler in ${ACE_SAMPLERS}; do
    run_name="${backbone}_${sampler}"
    echo "Generating ${run_name}"
    gen-mig-box-ace \
      "method=${backbone}+${sampler}" \
      "seed=[${ACE_SEEDS}]" \
      "target_levels=[${ACE_LEVELS}]" \
      "output_base_dir=${ACE_OUTPUT_ROOT}" \
      "run_name=${run_name}"
  done
done
