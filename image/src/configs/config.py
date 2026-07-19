"""Hydra registrations for the six Table E.10 reproduction runs."""

from hydra.core.config_store import ConfigStore

from configs.config_benchmark import MIGBenchConfig
from configs.config_method import (
    SD15ACEConfig,
    SD15FKCConfig,
    SD15NRConfig,
    SD21ACEConfig,
    SD21FKCConfig,
    SD21NRConfig,
)

config_store = ConfigStore.instance()
config_store.store(name="coco-mig", node=MIGBenchConfig)
config_store.store(group="method", name="sd15+nr", node=SD15NRConfig)
config_store.store(group="method", name="sd15+fkc", node=SD15FKCConfig)
config_store.store(group="method", name="sd15+ace", node=SD15ACEConfig)
config_store.store(group="method", name="sd21+nr", node=SD21NRConfig)
config_store.store(group="method", name="sd21+fkc", node=SD21FKCConfig)
config_store.store(group="method", name="sd21+ace", node=SD21ACEConfig)
