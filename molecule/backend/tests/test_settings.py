import pytest
from pydantic import ValidationError

from ace_backend.schemas import AvailableFeature, OperationalLimits, UnavailableFeature
from ace_backend.settings import Settings


@pytest.mark.parametrize(
    "limits",
    [
        {"max_num_samples": 0},
        {"max_num_samples": 17},
        {"max_num_samples": True},
        {"max_num_samples": "4"},
        {"max_running_jobs": 2},
        {"max_num_sampling_steps": 9},
        {"max_request_bytes": 1},
        {"unknown_limit": 1},
    ],
)
def test_invalid_limits_fail_at_configuration_boundary(limits: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        OperationalLimits.model_validate(limits)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:-1", "cuda:1;command"])
def test_invalid_device_is_rejected(device: str) -> None:
    with pytest.raises(ValidationError):
        Settings(device=device)


def test_unknown_disabled_feature_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACE_API_DISABLED_FEATURES", '["typo"]')
    with pytest.raises(ValidationError):
        Settings()


def test_response_states_require_consistent_reasons() -> None:
    with pytest.raises(ValidationError):
        AvailableFeature.model_validate({"available": True, "reason": "cuda_unavailable"})
    with pytest.raises(ValidationError):
        UnavailableFeature.model_validate({"available": False, "reason": None})
    with pytest.raises(ValidationError):
        OperationalLimits().max_num_samples = 3
