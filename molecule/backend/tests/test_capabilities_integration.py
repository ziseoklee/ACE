"""HTTP integration tests with real CPU dependencies and isolated GPU/file boundaries."""

import importlib
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest
import torch
from fastapi.testclient import TestClient

from ace_backend import capabilities as readiness
from ace_backend.app import create_app
from ace_backend.schemas import AvailableFeature, Capabilities, FeatureName, OperationalLimits, UnavailableFeature
from ace_backend.settings import Settings

ALL_FEATURES: frozenset[FeatureName] = frozenset({"inference", "druglikeness", "scaffold_preservation", "docking"})


def get_capabilities(settings: Settings | None = None) -> Capabilities:
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        return Capabilities.model_validate_json(response.content)


@pytest.fixture
def model_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, ...]:
    """Small readable files at the actual expert asset boundary; never deserialize weights."""
    from experts import diffsbdd_expert, edm_expert, geodiff_expert

    edm = replace(edm_expert.EDM_PRETRAINED_SPECS[edm_expert.EDM_PRETRAINED_GEOM_DRUG], output_dir=tmp_path)
    monkeypatch.setitem(edm_expert.EDM_PRETRAINED_SPECS, edm_expert.EDM_PRETRAINED_GEOM_DRUG, edm)
    monkeypatch.setattr(geodiff_expert, "GEODIFF_CKPT_PATH", tmp_path / "geodiff.pt")
    monkeypatch.setattr(geodiff_expert, "GEODIFF_CONFIG_PATH", tmp_path / "geodiff.yml")
    monkeypatch.setattr(diffsbdd_expert, "DIFFSBDD_CKPT_PATH", tmp_path / "diffsbdd.ckpt")
    paths = readiness.inference_checkpoint_paths()
    for path in paths:
        path.write_bytes(b"readiness fixture, not model weights\n")

    def forbid_model_loading(*args: object, **kwargs: object) -> None:
        raise AssertionError("A capabilities request must not deserialize model weights")

    monkeypatch.setattr(torch, "load", forbid_model_loading)
    monkeypatch.setattr(readiness, "check_cuda", lambda device: AvailableFeature())
    return paths


def test_no_cuda_keeps_real_cpu_evaluations_available() -> None:
    result = get_capabilities()
    assert result.inference == UnavailableFeature(reason="cuda_unavailable")
    assert result.evaluation.druglikeness == AvailableFeature()
    assert result.evaluation.scaffold_preservation == AvailableFeature()
    assert result.evaluation.docking == AvailableFeature()


def test_ready_response_matches_contract(model_assets: tuple[Path, ...]) -> None:
    result = get_capabilities()
    assert result.model_dump(mode="json") == {
        "api_version": "1.0.0",
        "inference": {"available": True, "reason": None},
        "evaluation": {
            "druglikeness": {"available": True, "reason": None},
            "scaffold_preservation": {"available": True, "reason": None},
            "docking": {"available": True, "reason": None},
        },
        "inference_presets": ["ace_scaffold_v1"],
        "limits": {
            "max_file_bytes": 10485760,
            "max_request_bytes": 33554432,
            "max_config_bytes": 16384,
            "max_pending_jobs": 4,
            "max_running_jobs": 1,
            "max_num_samples": 16,
            "max_num_sampling_steps": 2000,
            "max_num_ligand_atoms": 128,
            "max_pocket_atoms": 10000,
            "job_timeout_seconds": 3600,
        },
    }


@pytest.mark.parametrize("asset_index", range(5))
def test_missing_model_asset_disables_only_inference(model_assets: tuple[Path, ...], asset_index: int) -> None:
    model_assets[asset_index].unlink()
    result = get_capabilities()
    assert result.inference == UnavailableFeature(reason="checkpoint_missing")
    assert result.evaluation.druglikeness.available
    assert result.evaluation.docking.available


def test_empty_checkpoint_is_not_ready(model_assets: tuple[Path, ...]) -> None:
    model_assets[0].write_bytes(b"")
    assert get_capabilities().inference == UnavailableFeature(reason="checkpoint_missing")


def test_missing_inference_dependency_is_reported(
    monkeypatch: pytest.MonkeyPatch, model_assets: tuple[Path, ...]
) -> None:
    original_import = importlib.import_module

    def import_without_runtime(name: str, package: str | None = None) -> ModuleType:
        if name == "inference.condition_sampling":
            raise ImportError("An inference dependency is missing")
        return original_import(name, package)

    monkeypatch.setattr(readiness.importlib, "import_module", import_without_runtime)
    result = get_capabilities()
    assert result.inference == UnavailableFeature(reason="dependency_unavailable")
    assert result.evaluation.druglikeness.available


def test_missing_torch_does_not_prevent_http_or_cpu_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    result = get_capabilities()
    assert result.inference == UnavailableFeature(reason="dependency_unavailable")
    assert result.evaluation.druglikeness.available
    assert result.evaluation.scaffold_preservation.available
    assert result.evaluation.docking.available


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (RuntimeError("no kernel image is available for execution on the device"), "cuda_incompatible"),
        (torch.cuda.OutOfMemoryError("test allocation failure"), "cuda_unavailable"),
    ],
)
def test_cuda_kernel_failure_is_a_capability_not_an_http_error(
    monkeypatch: pytest.MonkeyPatch, failure: RuntimeError, reason: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "init", lambda: None)

    def fail_allocation(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(torch, "ones", fail_allocation)
    result = get_capabilities()
    assert not result.inference.available
    assert result.inference.reason == reason
    assert result.evaluation.druglikeness.available


def test_selected_device_must_be_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert get_capabilities(Settings(device="cuda:1")).inference == UnavailableFeature(reason="cuda_unavailable")


def test_missing_sa_scorer_disables_only_druglikeness(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation.metrics import druglikeness

    monkeypatch.setattr(druglikeness, "sascorer", None)
    result = get_capabilities()
    assert result.evaluation.druglikeness == UnavailableFeature(reason="dependency_unavailable")
    assert result.evaluation.scaffold_preservation.available
    assert result.evaluation.docking.available


def test_unreadable_sa_data_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation.metrics import druglikeness

    def fail_read() -> None:
        raise FileNotFoundError("missing SA fragment scores")

    monkeypatch.setattr(druglikeness.sascorer, "readFragmentScores", fail_read)
    result = get_capabilities()
    assert result.evaluation.druglikeness == UnavailableFeature(reason="dependency_unavailable")
    assert result.evaluation.scaffold_preservation.available


@pytest.mark.parametrize("tool", ["qvina", "obabel"])
def test_missing_docking_tool_keeps_other_evaluations_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool: str
) -> None:
    from evaluation.backends import qvina

    if tool == "qvina":
        monkeypatch.setattr(qvina, "PATH_QVINA2", tmp_path / "missing-qvina")
    else:
        monkeypatch.setenv("PATH", str(tmp_path))
    result = get_capabilities()
    assert result.evaluation.docking == UnavailableFeature(reason="dependency_unavailable")
    assert result.evaluation.druglikeness.available
    assert result.evaluation.scaffold_preservation.available


@pytest.mark.parametrize("script", ["#!/bin/sh\nexit 1\n", "not an executable format\n"])
def test_nonworking_executable_is_not_advertised(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str) -> None:
    from evaluation.backends import qvina

    executable = tmp_path / "qvina"
    executable.write_text(script)
    executable.chmod(0o700)
    monkeypatch.setattr(qvina, "PATH_QVINA2", executable)
    assert get_capabilities().evaluation.docking == UnavailableFeature(reason="dependency_unavailable")


def test_docking_probe_timeout_does_not_prevent_server_start(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(command: list[str], **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(command, 5)

    monkeypatch.setattr(readiness.subprocess, "run", timed_out)
    assert get_capabilities().evaluation.docking == UnavailableFeature(reason="dependency_unavailable")


def test_disabled_features_skip_probes_and_keep_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object) -> None:
        raise AssertionError("Disabled features must not probe dependencies or devices")

    for name in ("check_inference", "check_druglikeness", "check_scaffold_preservation", "check_docking"):
        monkeypatch.setattr(readiness, name, forbidden)
    result = get_capabilities(Settings(disabled_features=ALL_FEATURES))
    assert result.inference == UnavailableFeature(reason="feature_not_enabled")
    assert result.evaluation.druglikeness == UnavailableFeature(reason="feature_not_enabled")
    assert result.evaluation.scaffold_preservation == UnavailableFeature(reason="feature_not_enabled")
    assert result.evaluation.docking == UnavailableFeature(reason="feature_not_enabled")
    assert result.inference_presets == ("ace_scaffold_v1",)


def test_limits_and_feature_settings_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACE_API_DISABLED_FEATURES", '["inference", "docking"]')
    monkeypatch.setenv("ACE_API_LIMITS", '{"max_num_samples": 3, "max_pending_jobs": 2}')
    result = get_capabilities()
    assert result.inference == UnavailableFeature(reason="feature_not_enabled")
    assert result.evaluation.docking == UnavailableFeature(reason="feature_not_enabled")
    assert result.evaluation.druglikeness.available
    assert result.limits == OperationalLimits(max_num_samples=3, max_pending_jobs=2)


def test_get_reuses_startup_snapshot_and_issues_request_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(create_app()) as client:

        def forbidden(*args: object) -> None:
            raise AssertionError("GET must not reinitialize CUDA or reload dependencies")

        monkeypatch.setattr(readiness, "check_cuda", forbidden)
        monkeypatch.setattr(readiness, "check_druglikeness", forbidden)
        first = client.get("/api/v1/capabilities")
        second = client.get("/api/v1/capabilities")
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.headers["cache-control"] == "no-store"
    assert UUID(first.headers["x-request-id"]) != UUID(second.headers["x-request-id"])


def test_openapi_docs_and_cors() -> None:
    settings = Settings(disabled_features=ALL_FEATURES, cors_origins=("http://localhost:5173",))
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/capabilities", headers={"Origin": "http://localhost:5173"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
        assert "X-Request-ID" in response.headers["access-control-expose-headers"]
        assert (
            "access-control-allow-origin"
            not in client.get("/api/v1/capabilities", headers={"Origin": "https://unlisted.example"}).headers
        )
        preflight = client.options(
            "/api/v1/capabilities",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
        )
        assert preflight.status_code == 200
        UUID(preflight.headers["x-request-id"])
        assert client.get("/docs").status_code == 200
        specification = client.get("/openapi.json")
        assert specification.status_code == 200
        operation = specification.json()["paths"]["/api/v1/capabilities"]["get"]
        assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/Capabilities")


@pytest.mark.parametrize(
    ("method", "path", "status", "code"),
    [
        ("GET", "/capabilities", 404, "not_found"),
        ("POST", "/api/v1/capabilities", 405, "method_not_allowed"),
    ],
)
def test_routing_errors_follow_common_error_contract(method: str, path: str, status: int, code: str) -> None:
    with TestClient(create_app(Settings(disabled_features=ALL_FEATURES))) as client:
        response = client.request(method, path)
    assert response.status_code == status
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["details"] == []
    if status == 405:
        assert response.headers["allow"] == "GET"


def test_internal_errors_do_not_expose_server_details() -> None:
    app = create_app(Settings(disabled_features=ALL_FEATURES))

    @app.get("/test-failure")
    async def fail() -> None:
        raise RuntimeError("Private path /srv/model-secret/checkpoint.pt")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/test-failure")
    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "internal_error",
        "message": "An unexpected server error occurred.",
        "details": [],
    }
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert "model-secret" not in response.text
