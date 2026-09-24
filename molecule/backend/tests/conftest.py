import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_runtime_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep tests independent of deployment settings and physical GPU access."""
    for name in tuple(os.environ):
        if name.startswith("ACE_API_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
