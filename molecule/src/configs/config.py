from hydra.core.config_store import ConfigStore

from .config_benchmark import CrossDocked2020BenchConfig
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

# Rgister configurations with ConfigStore
cs = ConfigStore.instance()

# Benchmark configs
cs.store(group="benchmark", name="CrossDocked2020BenchConfig", node=CrossDocked2020BenchConfig)

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
