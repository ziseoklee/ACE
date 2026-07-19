"""Structured COCO-MIG configuration for ACE image reproduction."""

from dataclasses import dataclass, field
from typing import Any

from configs.config_method import BaseMethodConfig


@dataclass
class MIGBenchConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_", {"method": "sd21+ace"}])
    seed: list[int] = field(default_factory=list)
    name: str = "COCO-MIG Benchmark"
    target_levels: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    dataset_base_dir: str = "data"
    dataset_jsonl_path: str = "data/mig_bench.jsonl"
    output_base_dir: str = "output/COCO-MIG_Bench"
    run_name: str | None = None
    mode: str = "accuracy"
    method: BaseMethodConfig = field(default_factory=BaseMethodConfig)

    def __post_init__(self) -> None:
        if self.mode == "accuracy":
            if not self.seed:
                self.seed = [42, 37, 519, 609, 123, 401, 780, 0]
        elif self.mode == "efficiency":
            if not self.seed:
                self.seed = [42]
        else:
            raise ValueError(f"mode must be 'accuracy' or 'efficiency', got {self.mode!r}.")
