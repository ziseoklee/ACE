from hydra.core.config_store import ConfigStore

from .config_benchmark import CrossDocked2020BenchConfig
from .config_moe import MoEConfig
from .config_moe_component import (
    DIFFSBDD_CROSSDOCKED_FULLATOM_COND,
    EDM_GEOM_DRUG,
    EDM_GEOM_DRUG_LIGAND,
    EDM_QM9_FRAGMENT,
    EDM_QM9_LIGAND,
    GEODIFF_QM9_FRAGMENT,
)
from .config_sampler import ACEliteSamplerConfig, ACESamplerConfig, FKCSamplerConfig, NRSamplerConfig
from .config_weight import (
    ACEBumpWeightConfig,
    ConstantWeightConfig,
    LambdaBumpWeightConfig,
    LinearDecreasingWeightConfig,
    LinearIncreasingWeightConfig,
    QuadraticBumpWeightConfig,
    VBumpWeightConfig,
)

# Register configurations with ConfigStore
cs = ConfigStore.instance()

# Benchmark configs
cs.store(group="benchmark", name="CrossDocked2020BenchConfig", node=CrossDocked2020BenchConfig)

# MoE composition configs
cs.store(group="moe", name="DefaultMoE", node=MoEConfig)

# Sampler configs
cs.store(group="sampler", name="NRSampler", node=NRSamplerConfig)
cs.store(group="sampler", name="FKCSampler", node=FKCSamplerConfig)
cs.store(group="sampler", name="ACESampler", node=ACESamplerConfig)
cs.store(group="sampler", name="ACEliteSampler", node=ACEliteSamplerConfig)

# Weight configs
cs.store(group="weight", name="ConstantWeight", node=ConstantWeightConfig)
cs.store(group="weight", name="LinearIncreasingWeight", node=LinearIncreasingWeightConfig)
cs.store(group="weight", name="LinearDecreasingWeight", node=LinearDecreasingWeightConfig)
cs.store(group="weight", name="LambdaBumpWeight", node=LambdaBumpWeightConfig)
cs.store(group="weight", name="VBumpWeight", node=VBumpWeightConfig)
cs.store(group="weight", name="QuadraticBumpWeight", node=QuadraticBumpWeightConfig)
cs.store(group="weight", name="ACEBumpWeight", node=ACEBumpWeightConfig)

# MoE component configs
cs.store(group="moe_component", name="EDM_QM9_FRAGMENT", node=EDM_QM9_FRAGMENT)
cs.store(group="moe_component", name="EDM_QM9_LIGAND", node=EDM_QM9_LIGAND)
cs.store(group="moe_component", name="EDM_GEOM_DRUG", node=EDM_GEOM_DRUG)
cs.store(group="moe_component", name="EDM_GEOM_DRUG_LIGAND", node=EDM_GEOM_DRUG_LIGAND)
cs.store(group="moe_component", name="GEODIFF_QM9_FRAGMENT", node=GEODIFF_QM9_FRAGMENT)
cs.store(group="moe_component", name="DIFFSBDD_CROSSDOCKED_FULLATOM_COND", node=DIFFSBDD_CROSSDOCKED_FULLATOM_COND)
