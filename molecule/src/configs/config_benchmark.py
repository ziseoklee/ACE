from dataclasses import dataclass
from pathlib import Path


@dataclass
class CrossDocked2020BenchConfig:
    name: str = "CrossDocked2020"
    data_root: str = str(Path(__file__).parents[2] / "data" / "crossdocked")
    save_dir: str = str(Path(__file__).parents[2] / "outputs" / "crossdocked2020")
    num_trials: int = 1  # Number of times to repeat the inference process for each configuration
    seed: int = 42
