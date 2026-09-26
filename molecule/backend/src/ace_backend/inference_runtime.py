"""The fixed v1 preset, using the existing ACE configs and sampling implementation."""

from typing import TYPE_CHECKING

from ace_backend.jobs_schema import InferenceConfig

if TYPE_CHECKING:
    from configs.config_moe import MoEConfig
    from configs.config_sampler import ACESamplerConfig


def build_preset(config: InferenceConfig, device: str) -> tuple["ACESamplerConfig", "MoEConfig"]:
    from configs.config_moe import MoEConfig, MoEExponentConfig
    from configs.config_sampler import ACESamplerConfig
    from configs.config_weight import ACEBumpWeightConfig, ConstantWeightConfig
    from pipelines.diffsbdd.components import DIFFSBDD_CROSSDOCKED_FULLATOM_COND
    from pipelines.edm.components import EDM_GEOM_DRUG_FRAGMENT, EDM_GEOM_DRUG_LIGAND
    from pipelines.geodiff.components import GEODIFF_QM9_FRAGMENT

    sampler = ACESamplerConfig(
        use_logq=True,
        do_resample=True,
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
        omega=config.ace.omega,
        global_scheduler_key="GEODIFF",
        diffusion_scale=config.ace.diffusion_scale,
        components={
            "edm_fragment": EDM_GEOM_DRUG_FRAGMENT,
            "edm_ligand": EDM_GEOM_DRUG_LIGAND,
            "geodiff_fragment": GEODIFF_QM9_FRAGMENT,
            "diffsbdd": DIFFSBDD_CROSSDOCKED_FULLATOM_COND,
        },
        exponents={
            "edm_fragment": MoEExponentConfig(ConstantWeightConfig(config.ace.omega), weight_scale=-1.0),
            "edm_ligand": MoEExponentConfig(ConstantWeightConfig(config.ace.omega), weight_scale=-1.0, constant=1.0),
            "geodiff_fragment": MoEExponentConfig(ConstantWeightConfig(config.ace.omega)),
            "diffsbdd": MoEExponentConfig(ACEBumpWeightConfig(config.ace.omega, B1=config.ace.b1, B2=config.ace.b2)),
        },
    )
    return sampler, moe
