#!/usr/bin/env bash
set -euo pipefail

OMEGA="${OMEGA:-1.4}"
DEVICE="${DEVICE:-cuda:0}"
NUM_LIGAND_ATOMS="${NUM_LIGAND_ATOMS:-25}"
SEED="${SEED:-42}"
B1="${B1:-30.0}"
B2="${B2:-0.336}"


printf "Example sampling script for MoE inference with CLI overrides.\n"
printf "======================================================\n"
printf "[ Part 1: Sampling with single experts ]\n\n"

printf "Running de-novo ligand generation with EDM_GEOM_DRUG_LIGAND expert only...\n"
uv run ace-infer \
    --config-name inference_base \
    "sampler=NRSampler" \
    "sampler.seed=${SEED}" \
    "sampler.device=${DEVICE}" \
    "moe.omega=1.0" \
    "moe.global_scheduler_key=EDM" \
    "moe.diffusion_scale=1.0" \
    "+moe_component@moe.components.edm=EDM_GEOM_DRUG_LIGAND" \
    "+weight@moe.exponents.edm.weight_fn=ConstantWeight" \
    "moe.exponents.edm.weight_fn.omega=1.0" \
    "moe.exponents.edm.weight_scale=1.0" \
    "moe.exponents.edm.constant=0.0" \
    "data.num_ligand_atoms=${NUM_LIGAND_ATOMS}"

printf "Running conformer generation with GEODIFF_QM9_FRAGMENT expert only...\n"
uv run ace-infer \
    --config-name inference_base \
    "sampler=NRSampler" \
    "sampler.seed=${SEED}" \
    "sampler.device=${DEVICE}" \
    "moe.omega=1.0" \
    "moe.global_scheduler_key=GEODIFF" \
    "moe.diffusion_scale=1.0" \
    "+moe_component@moe.components.geodiff=GEODIFF_QM9_FRAGMENT" \
    "+weight@moe.exponents.geodiff.weight_fn=ConstantWeight" \
    "moe.exponents.geodiff.weight_fn.omega=1.0" \
    "moe.exponents.geodiff.weight_scale=1.0" \
    "moe.exponents.geodiff.constant=0.0" \
    "data.num_ligand_atoms=11"  # number of fragment atoms of the default input (examples/4m7t_fragment.sdf)

printf "Running structure-based drug design (SBDD) with DIFFSBDD_CROSSDOCKED_FULLATOM expert only...\n"
uv run ace-infer \
    --config-name inference_base \
    "sampler=NRSampler" \
    "sampler.seed=${SEED}" \
    "sampler.device=${DEVICE}" \
    "moe.omega=1.0" \
    "moe.global_scheduler_key=DIFFSBDD" \
    "moe.diffusion_scale=1.0" \
    "+moe_component@moe.components.diffsbdd=DIFFSBDD_CROSSDOCKED_FULLATOM_COND" \
    "+weight@moe.exponents.diffsbdd.weight_fn=ConstantWeight" \
    "moe.exponents.diffsbdd.weight_fn.omega=1.0" \
    "moe.exponents.diffsbdd.weight_scale=1.0" \
    "moe.exponents.diffsbdd.constant=0.0" \
    "data.num_ligand_atoms=${NUM_LIGAND_ATOMS}"

printf "\n======================================================\n"


printf "\n======================================================\n"
printf "[ Part 2: Applying guidance with de-novo expert and task-specific experts ]\n\n"

printf "Running guided conformer generation with EDM_GEOM_DRUG_FRAGMENT (DN) and GEODIFF_QM9_FRAGMENT (CONF) experts...\n"
uv run ace-infer \
    --config-name inference_base \
    "sampler=NRSampler" \
    "sampler.seed=${SEED}" \
    "sampler.device=${DEVICE}" \
    "moe.omega=${OMEGA}" \
    "moe.global_scheduler_key=GEODIFF" \
    "moe.diffusion_scale=1.0" \
    "+moe_component@moe.components.geodiff_fragment=GEODIFF_QM9_FRAGMENT" \
    "+weight@moe.exponents.geodiff_fragment.weight_fn=ConstantWeight" \
    "moe.exponents.geodiff_fragment.weight_fn.omega=${OMEGA}" \
    "moe.exponents.geodiff_fragment.weight_scale=1.0" \
    "moe.exponents.geodiff_fragment.constant=0.0" \
    "+moe_component@moe.components.edm_fragment=EDM_GEOM_DRUG_FRAGMENT" \
    "+weight@moe.exponents.edm_fragment.weight_fn=ConstantWeight" \
    "moe.exponents.edm_fragment.weight_fn.omega=${OMEGA}" \
    "moe.exponents.edm_fragment.weight_scale=-1.0" \
    "moe.exponents.edm_fragment.constant=1.0" \
    "data.num_ligand_atoms=11"  # number of fragment atoms of the default input (examples/4m7t_fragment.sdf)

printf "Running guided structure-based drug design (SBDD) with EDM_GEOM_DRUG_LIGAND (DN) and DIFFSBDD_CROSSDOCKED_FULLATOM (SBDD) experts...\n"
uv run ace-infer \
    --config-name inference_base \
    "sampler=NRSampler" \
    "sampler.seed=${SEED}" \
    "sampler.device=${DEVICE}" \
    "moe.omega=${OMEGA}" \
    "moe.global_scheduler_key=DIFFSBDD" \
    "moe.diffusion_scale=2.0" \
    "+moe_component@moe.components.diffsbdd=DIFFSBDD_CROSSDOCKED_FULLATOM_COND" \
    "+weight@moe.exponents.diffsbdd.weight_fn=ConstantWeight" \
    "moe.exponents.diffsbdd.weight_fn.omega=${OMEGA}" \
    "moe.exponents.diffsbdd.weight_scale=1.0" \
    "moe.exponents.diffsbdd.constant=0.0" \
    "+moe_component@moe.components.edm=EDM_GEOM_DRUG_LIGAND" \
    "+weight@moe.exponents.edm.weight_fn=ConstantWeight" \
    "moe.exponents.edm.weight_fn.omega=${OMEGA}" \
    "moe.exponents.edm.weight_scale=-1.0" \
    "moe.exponents.edm.constant=1.0" \
    "data.num_ligand_atoms=${NUM_LIGAND_ATOMS}"

printf "\n======================================================\n"


printf "\n======================================================\n"
printf "[ Part 3: Applying flexible-pose scaffold decoration with four experts (DN_fragment, DN_ligand, CONF, SBDD)]\n\n"

printf "Running guided flexible-pose scaffold decoration with EDM_GEOM_DRUG_FRAGMENT (DN_fragment), EDM_GEOM_DRUG_LIGAND (DN_ligand), GEODIFF_QM9_FRAGMENT (CONF), and DIFFSBDD_CROSSDOCKED_FULLATOM (SBDD) experts...\n"

# the following command is equivalent to `inference` config except that it uses `ACEliteSampler` for multi-expert sampling instead of `ACESampler``
uv run ace-infer \
    --config-name inference_base \
    "sampler=ACEliteSampler" \
    "sampler.seed=${SEED}" \
    "sampler.device=${DEVICE}" \
    "moe.omega=${OMEGA}" \
    "moe.global_scheduler_key=GEODIFF" \
    "moe.diffusion_scale=2.0" \
    "+moe_component@moe.components.edm_fragment=EDM_GEOM_DRUG_FRAGMENT" \
    "+weight@moe.exponents.edm_fragment.weight_fn=ConstantWeight" \
    "moe.exponents.edm_fragment.weight_fn.omega=${OMEGA}" \
    "moe.exponents.edm_fragment.weight_scale=-1.0" \
    "moe.exponents.edm_fragment.constant=0.0" \
    "+moe_component@moe.components.edm_ligand=EDM_GEOM_DRUG_LIGAND" \
    "+weight@moe.exponents.edm_ligand.weight_fn=ConstantWeight" \
    "moe.exponents.edm_ligand.weight_fn.omega=${OMEGA}" \
    "moe.exponents.edm_ligand.weight_scale=-1.0" \
    "moe.exponents.edm_ligand.constant=1.0" \
    "+moe_component@moe.components.geodiff_fragment=GEODIFF_QM9_FRAGMENT" \
    "+weight@moe.exponents.geodiff_fragment.weight_fn=ConstantWeight" \
    "moe.exponents.geodiff_fragment.weight_fn.omega=${OMEGA}" \
    "moe.exponents.geodiff_fragment.weight_scale=1.0" \
    "moe.exponents.geodiff_fragment.constant=0.0" \
    "+moe_component@moe.components.diffsbdd=DIFFSBDD_CROSSDOCKED_FULLATOM_COND" \
    "+weight@moe.exponents.diffsbdd.weight_fn=ACEBumpWeight" \
    "moe.exponents.diffsbdd.weight_fn.omega=${OMEGA}" \
    "moe.exponents.diffsbdd.weight_scale=1.0" \
    "moe.exponents.diffsbdd.constant=0.0" \
    "moe.exponents.diffsbdd.weight_fn.B1=${B1}" \
    "moe.exponents.diffsbdd.weight_fn.B2=${B2}" \
    "data.num_ligand_atoms=${NUM_LIGAND_ATOMS}"

printf "\n======================================================\n"

printf "All done! Check the output files.\n"
