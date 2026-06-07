from dataclasses import dataclass
from pathlib import Path


@dataclass
class CrossDocked2020BenchConfig:
    name: str = "CrossDocked2020"
    data_root: str = str(Path(__file__).parents[2] / "data" / "crossdocked")
    num_trials: int = 1  # Number of times to repeat the inference process for each configuration
    seed: int = 42  # Random seed for reproducibility
