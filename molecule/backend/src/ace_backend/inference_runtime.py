"""Fixed v1 presets, using the existing configs and sampling implementation."""

from typing import TYPE_CHECKING

from ace_backend.jobs_schema import ACEParameters, ConstantMoEParameters, InferenceConfig

if TYPE_CHECKING:
    from configs.config_moe import MoEConfig
    from configs.config_sampler import _BaseSamplerConfig


def build_preset(config: InferenceConfig, device: str) -> tuple["_BaseSamplerConfig", "MoEConfig"]:
    from configs.config_moe import MoEConfig, MoEExponentConfig
    from configs.config_sampler import ACESamplerConfig, FKCSamplerConfig, NRSamplerConfig
    from configs.config_weight import ACEBumpWeightConfig, ConstantWeightConfig
    from pipelines.diffsbdd.components import DIFFSBDD_CROSSDOCKED_FULLATOM_COND
    from pipelines.edm.components import EDM_GEOM_DRUG_FRAGMENT, EDM_GEOM_DRUG_LIGAND
    from pipelines.geodiff.components import GEODIFF_QM9_FRAGMENT

    sampler_type = {
        "nr_scaffold_v1": NRSamplerConfig,
        "fkc_scaffold_v1": FKCSamplerConfig,
        "ace_scaffold_v1": ACESamplerConfig,
    }[config.preset]
    parameters: ACEParameters | ConstantMoEParameters
    diffsbdd_weight: ACEBumpWeightConfig | ConstantWeightConfig
    if config.preset == "ace_scaffold_v1":
        parameters = config.ace
        diffsbdd_weight = ACEBumpWeightConfig(parameters.omega, B1=parameters.b1, B2=parameters.b2)
    else:
        parameters = config.moe
        diffsbdd_weight = ConstantWeightConfig(parameters.omega)

    sampler = sampler_type(
        use_logq=config.preset == "ace_scaffold_v1",
        do_resample=config.preset != "nr_scaffold_v1",
        dlogq_calc_interval=10,
        dlogq_noise_scale=3.16227766017,
        resampling_step_interval=10,
        ode_start_t=0.98,
        batch_size=config.num_samples,
        num_sampling_steps=config.num_sampling_steps,
        seed=config.seed,
        device=device,
    )
    moe = MoEConfig(
        omega=parameters.omega,
        global_scheduler_key="GEODIFF",
        diffusion_scale=parameters.diffusion_scale,
        components={
            "edm_fragment": EDM_GEOM_DRUG_FRAGMENT,
            "edm_ligand": EDM_GEOM_DRUG_LIGAND,
            "geodiff_fragment": GEODIFF_QM9_FRAGMENT,
            "diffsbdd": DIFFSBDD_CROSSDOCKED_FULLATOM_COND,
        },
        exponents={
            "edm_fragment": MoEExponentConfig(ConstantWeightConfig(parameters.omega), weight_scale=-1.0),
            "edm_ligand": MoEExponentConfig(ConstantWeightConfig(parameters.omega), weight_scale=-1.0, constant=1.0),
            "geodiff_fragment": MoEExponentConfig(ConstantWeightConfig(parameters.omega)),
            "diffsbdd": MoEExponentConfig(diffsbdd_weight),
        },
    )
    return sampler, moe
