"""Validated server configuration; never supplied by API clients."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ace_backend.schemas import FeatureName, OperationalLimits


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ACE_API_", frozen=True, extra="forbid")

    device: str = Field(default="cuda:0", pattern=r"^cuda:(0|[1-9][0-9]*)$")
    disabled_features: frozenset[FeatureName] = frozenset()
    cors_origins: tuple[str, ...] = ()
    storage_dir: Path = Path("outputs/web_demo")
    limits: OperationalLimits = Field(default_factory=OperationalLimits)
