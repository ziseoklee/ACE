"""Evaluation through real multipart, scientific inputs, subprocesses, storage, and retrieval."""

import copy
import hashlib
import json
import math
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import httpx2
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem
from test_inference_integration import ROOT, FileParts, assert_error, molecule_sdf, parts, replace_part, wait_status

import ace_backend.app as api
import ace_backend.dispatcher as dispatch
from ace_backend.evaluation_schema import EVALUATION_CONFIG_EXAMPLE
from ace_backend.schemas import (
    AvailableFeature,
    Capabilities,
    EvaluationCapabilities,
    OperationalLimits,
    UnavailableFeature,
)
from ace_backend.settings import Settings
from evaluation.backends.qvina import PATH_QVINA2

POST_URL = "/api/v1/evaluation/jobs"
DOCKING = {"seed": 42, "exhaustiveness": 1, "num_modes": 1, "padding_angstrom": 8.0}


def evaluation_parts(config: dict[str, object] | None = None, *, ligand: bytes | None = None) -> FileParts:
    config = EVALUATION_CONFIG_EXAMPLE if config is None else config
    result: FileParts = [("config", (None, json.dumps(config), "application/json"))]
    if config["source"]["type"] == "upload":
        result.append(
            (
                "ligand_sdf",
                (
                    "ligand.sdf",
                    ligand if ligand is not None else (ROOT / "examples/4m7t_ligand.sdf").read_bytes(),
                    "application/octet-stream",
                ),
            )
        )
        needed = set()
        if "scaffold_preservation" in config["metrics"]:
            needed.add("fragment_sdf")
        if "docking" in config["metrics"]:
            needed.update(("pocket_pdb", "reference_ligand_sdf"))
        result.extend(part for part in parts() if part[0] in needed)
    return result


@pytest.fixture
def make_evaluation_client(monkeypatch: pytest.MonkeyPatch) -> Callable[..., TestClient]:
    def readiness(settings: Settings) -> Capabilities:
        return Capabilities(
            inference=UnavailableFeature(reason="feature_not_enabled")
            if "inference" in settings.disabled_features
            else AvailableFeature(),
            evaluation=EvaluationCapabilities(
                **{
                    metric: UnavailableFeature(reason="feature_not_enabled")
                    if metric in settings.disabled_features
                    else AvailableFeature()
                    for metric in ("druglikeness", "scaffold_preservation", "docking")
                }
            ),
            limits=settings.limits,
        )

    monkeypatch.setattr(api, "detect_capabilities", readiness)

    def factory(*, disabled: frozenset[str] = frozenset(), **limits: int) -> TestClient:
        return TestClient(
            api.create_app(Settings(disabled_features=disabled, limits=OperationalLimits(**limits))),
            raise_server_exceptions=False,
        )

    return factory


def block_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dispatch, "worker_command", lambda root, device: (sys.executable, "-c", "import time; time.sleep(600)")
    )


def evaluation_worker_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(
        dispatch,
        "worker_command",
        lambda root, device: (
            sys.executable,
            str(Path(__file__).with_name("fake_evaluation_worker.py")),
            str(root),
            mode,
        ),
    )


def completed_result(client: TestClient, job: dict[str, object]) -> dict[str, object]:
    final = wait_status(client, job["links"]["self"], {"succeeded", "failed"}, timeout=60)
    assert final["status"] == "succeeded", final
    response = client.get(job["links"]["result"])
    assert response.status_code == 200, response.text
    return response.json()


def check_artifacts(client: TestClient, job: dict[str, object]) -> list[dict[str, object]]:
    response = client.get(job["links"]["artifacts"])
    assert response.status_code == 200
    manifest = response.json()["artifacts"]
    for artifact in manifest:
        fetched = client.get(artifact["url"] + "?download=true")
        assert fetched.status_code == 200
        assert fetched.headers["Content-Disposition"].startswith("attachment;")
        assert fetched.headers["Cache-Control"] == "no-store"
        assert hashlib.sha256(fetched.content).hexdigest() == artifact["sha256"]
        assert len(fetched.content) == artifact["size_bytes"]
    return manifest


def test_upload_runs_real_cpu_metrics_and_survives_restart(make_evaluation_client: Callable[..., TestClient]) -> None:
    ligand = molecule_sdf("CCOc1ccccc1", is_3d=False)
    config = {"source": {"type": "upload"}, "metrics": ["druglikeness", "scaffold_preservation"], "docking": None}
    form = replace_part(evaluation_parts(config, ligand=ligand), "fragment_sdf", molecule_sdf("c1ccccc1", is_3d=False))
    with make_evaluation_client(disabled=frozenset({"inference", "docking"})) as client:
        response = client.post(POST_URL, files=form)
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["kind"] == "evaluation" and job["status"] == "queued"
        assert response.headers["Location"] == job["links"]["self"]
        assert response.headers["Retry-After"] == "2"
        result = completed_result(client, job)
        assert result["source"] == config["source"] and result["metrics"] == config["metrics"]
        assert result["warnings"] == []
        groups = result["samples"][0]["metrics"]
        assert set(groups) == set(config["metrics"])
        assert all(report["status"] == "succeeded" for report in groups["druglikeness"].values())
        assert groups["druglikeness"]["validity"]["value"] is True
        assert groups["scaffold_preservation"]["value"] == {
            "contains_fragment": True,
            "method": "rdkit_substructure_v1",
        }
        artifacts = check_artifacts(client, job)
        stored_ligand = next(
            item for item in artifacts if item["role"] == "input_original" and item["filename"] == "ligand_sdf.sdf"
        )
        assert client.get(stored_ligand["url"]).content == ligand
        provenance = client.get(next(item["url"] for item in artifacts if item["role"] == "provenance")).json()
        assert provenance["environment"]["device"] == "cpu"
        assert "torch" not in provenance["environment"]["dependencies"]
        assert provenance["code"]["repositories"][0]["commit"]
        assert provenance["results"] == result
    with make_evaluation_client() as restarted:
        assert restarted.get(job["links"]["self"]).json()["status"] == "succeeded"
        assert restarted.get(job["links"]["result"]).json() == result
        assert restarted.get(stored_ligand["url"]).content == ligand


@pytest.mark.parametrize("filename", ["inference-config-nr.json", "inference-config-fkc.json", "inference-config.json"])
def test_source_snapshot_preserves_selected_samples_after_source_removal(
    make_evaluation_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, filename: str
) -> None:
    real_command = dispatch.worker_command

    def command(root: Path, device: str) -> tuple[str, ...]:
        if json.loads((root / "job.json").read_text())["kind"] == "inference":
            return (
                sys.executable,
                str(Path(__file__).with_name("fake_inference_worker.py")),
                str(root),
                device,
                "success",
            )
        return real_command(root, device)

    monkeypatch.setattr(dispatch, "worker_command", command)
    with make_evaluation_client() as client:
        inference_config = json.loads((ROOT / "examples" / filename).read_text())
        inference_config["num_samples"] = 3
        inference = client.post("/api/v1/inference/jobs", files=parts(config=inference_config)).json()
        inference_result = completed_result(client, inference)
        original = {item["artifact_id"]: client.get(item["url"]).content for item in check_artifacts(client, inference)}
        config = {
            "source": {"type": "inference_job", "job_id": inference["job_id"], "sample_ids": [2, 0]},
            "metrics": ["druglikeness", "scaffold_preservation"],
            "docking": None,
        }
        for ids in ([0, 1], [0, 99]):
            invalid = {**config, "source": {**config["source"], "sample_ids": ids}}
            assert_error(
                client.post(POST_URL, files=evaluation_parts(invalid)), 422, "validation_error", "invalid_sample"
            )
        assert len(list((tmp_path / "jobs").glob("*/job.json"))) == 1
        response = client.post(POST_URL, files=evaluation_parts(config))
        assert response.status_code == 202, response.text
        evaluation = response.json()
        assert client.get(inference["links"]["result"]).json() == inference_result
        shutil.rmtree(tmp_path / "jobs" / inference["job_id"])
        result = completed_result(client, evaluation)
        assert result["source"] == config["source"]
        assert [sample["sample_id"] for sample in result["samples"]] == [0, 2]
        manifest = check_artifacts(client, evaluation)
        copied = [item for item in manifest if item["role"] == "ligand"]
        assert len(copied) == 2
        assert all(client.get(item["url"]).content in original.values() for item in copied)
        assert len([item for item in manifest if item["role"] == "provenance"]) == 2
        assert_error(client.get(inference["links"]["self"]), 404, "job_not_found")
    with make_evaluation_client() as restarted:
        assert restarted.get(evaluation["links"]["result"]).json() == result
        check_artifacts(restarted, evaluation)


@pytest.mark.parametrize(
    "changes",
    [
        {"metrics": []},
        {"metrics": ["druglikeness", "druglikeness"]},
        {"metrics": ["unknown"]},
        {"metrics": "druglikeness"},
        {"metrics": [True]},
        {"docking": DOCKING},
        {"metrics": ["docking"]},
        {"source": {"type": "unknown"}},
        {"source": {"type": "upload", "path": "/private"}},
        {"extra": 1},
        {"metrics": None},
        {"source": None},
    ],
)
def test_invalid_configs(
    make_evaluation_client: Callable[..., TestClient], changes: dict[str, object], tmp_path: Path
) -> None:
    config = {**EVALUATION_CONFIG_EXAMPLE, **changes}
    with make_evaluation_client() as client:
        form = replace_part(evaluation_parts(), "config", json.dumps(config))
        assert_error(client.post(POST_URL, files=form), 422, "validation_error")
    assert not list((tmp_path / "jobs").glob("*/job.json"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", -1),
        ("seed", 2147483648),
        ("seed", True),
        ("seed", "42"),
        ("exhaustiveness", 0),
        ("exhaustiveness", 33),
        ("num_modes", 0),
        ("num_modes", 21),
        ("padding_angstrom", 0),
        ("padding_angstrom", 21),
        ("padding_angstrom", float("nan")),
        ("padding_angstrom", float("inf")),
        ("padding_angstrom", True),
        ("extra", 1),
    ],
)
def test_invalid_docking_config(make_evaluation_client: Callable[..., TestClient], field: str, value: object) -> None:
    config = {"source": {"type": "upload"}, "metrics": ["docking"], "docking": {**DOCKING, field: value}}
    with make_evaluation_client() as client:
        assert_error(client.post(POST_URL, files=evaluation_parts(config)), 422, "validation_error")


@pytest.mark.parametrize("field", ["source", "metrics", "docking"])
def test_all_config_fields_are_required(make_evaluation_client: Callable[..., TestClient], field: str) -> None:
    config = copy.deepcopy(EVALUATION_CONFIG_EXAMPLE)
    del config[field]
    with make_evaluation_client() as client:
        assert_error(
            client.post(POST_URL, files=replace_part(evaluation_parts(), "config", json.dumps(config))),
            422,
            "validation_error",
        )


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        ("{", 400),
        ('{"source":{"type":"upload","type":"upload"},"metrics":["druglikeness"],"docking":null}', 422),
        ("null", 422),
        (b"\xff", 400),
    ],
)
def test_bad_config_encoding_and_duplicates(
    make_evaluation_client: Callable[..., TestClient], raw: bytes | str, status: int
) -> None:
    with make_evaluation_client() as client:
        assert_error(
            client.post(POST_URL, files=replace_part(evaluation_parts(), "config", raw)),
            status,
            "malformed_request" if status == 400 else "validation_error",
        )


@pytest.mark.parametrize(
    "case", ["missing_config", "missing_ligand", "unnecessary", "unknown", "duplicate", "config_file", "ligand_field"]
)
def test_rejects_invalid_multipart_parts(make_evaluation_client: Callable[..., TestClient], case: str) -> None:
    form = evaluation_parts()
    if case.startswith("missing_"):
        form = [item for item in form if item[0] != ("config" if case == "missing_config" else "ligand_sdf")]
    elif case in {"unnecessary", "unknown"}:
        form.append(
            (
                "fragment_sdf" if case == "unnecessary" else "extra",
                ("extra.sdf", molecule_sdf("C"), "application/octet-stream"),
            )
        )
    elif case == "duplicate":
        form.append(form[1])
    elif case == "config_file":
        form = replace_part(form, "config", json.dumps(EVALUATION_CONFIG_EXAMPLE), "config.json")
    else:
        form[1] = ("ligand_sdf", (None, "not a file", "text/plain"))
    with make_evaluation_client() as client:
        assert_error(client.post(POST_URL, files=form), 422, "validation_error")


@pytest.mark.parametrize("case", ["file", "config", "body", "stream"])
def test_upload_byte_limits(make_evaluation_client: Callable[..., TestClient], case: str) -> None:
    limits = (
        {"max_file_bytes": 16}
        if case == "file"
        else {"max_config_bytes": 16}
        if case == "config"
        else {"max_request_bytes": 1000, "max_file_bytes": 1000, "max_config_bytes": 500}
    )
    request = httpx2.Request("POST", "http://testserver" + POST_URL, files=evaluation_parts())
    body = request.read()
    headers = dict(request.headers)
    if case == "stream":
        headers.pop("content-length")
    with make_evaluation_client(**limits) as client:
        assert_error(
            client.post(POST_URL, content=iter([body]) if case == "stream" else body, headers=headers),
            413,
            "payload_too_large",
        )


@pytest.mark.parametrize(
    ("content", "detail"),
    [(b"", "invalid_structure"), (b"invalid", "invalid_structure"), (molecule_sdf("CC") * 2, "multiple_records")],
)
def test_unparseable_ligands(make_evaluation_client: Callable[..., TestClient], content: bytes, detail: str) -> None:
    with make_evaluation_client() as client:
        assert_error(client.post(POST_URL, files=evaluation_parts(ligand=content)), 422, "validation_error", detail)


def test_docking_requires_3d_and_single_protein_model(make_evaluation_client: Callable[..., TestClient]) -> None:
    config = {"source": {"type": "upload"}, "metrics": ["docking"], "docking": DOCKING}
    with make_evaluation_client() as client:
        for field in ("ligand_sdf", "reference_ligand_sdf"):
            form = replace_part(evaluation_parts(config), field, molecule_sdf("CC", is_3d=False))
            assert_error(client.post(POST_URL, files=form), 422, "validation_error", "coordinates_required")
        form = evaluation_parts(config)
        pdb = next(item[1][1] for item in form if item[0] == "pocket_pdb")
        assert_error(
            client.post(
                POST_URL,
                files=replace_part(
                    form, "pocket_pdb", b"MODEL        1\n" + pdb + b"ENDMDL\nMODEL        2\n" + pdb + b"ENDMDL\n"
                ),
            ),
            422,
            "validation_error",
            "multiple_models",
        )


def test_missing_and_unavailable_metrics_are_rejected(
    make_evaluation_client: Callable[..., TestClient], tmp_path: Path
) -> None:
    config = {"source": {"type": "upload"}, "metrics": ["druglikeness", "docking"], "docking": DOCKING}
    with make_evaluation_client(disabled=frozenset({"docking"})) as client:
        assert_error(client.post(POST_URL, files=evaluation_parts(config)), 503, "evaluation_unavailable")
        assert client.get("/api/v1/capabilities").status_code == 200
    assert not list((tmp_path / "jobs").glob("*/job.json"))


def test_queue_restart_and_result_conflicts_are_shared(
    make_evaluation_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    block_workers(monkeypatch)
    with make_evaluation_client(max_pending_jobs=1) as client:
        inference = client.post("/api/v1/inference/jobs", files=parts()).json()
        wait_status(client, inference["links"]["self"], {"running"})
        evaluation = client.post(POST_URL, files=evaluation_parts()).json()
        assert_error(client.get(evaluation["links"]["result"]), 409, "result_not_ready")
        assert_error(client.post(POST_URL, files=evaluation_parts()), 429, "queue_full")
    with make_evaluation_client() as restarted:
        for job in (inference, evaluation):
            final = restarted.get(job["links"]["self"]).json()
            assert final["status"] == "failed" and final["error"]["code"] == "service_restarted"
            assert_error(restarted.get(job["links"]["result"]), 409, "job_failed")
        source = {
            "source": {"type": "inference_job", "job_id": inference["job_id"], "sample_ids": [0]},
            "metrics": ["druglikeness"],
            "docking": None,
        }
        assert_error(restarted.post(POST_URL, files=evaluation_parts(source)), 409, "source_job_not_succeeded")


@pytest.mark.parametrize(
    ("mode", "code"),
    [("crash", "evaluation_failed"), ("timeout", "job_timeout"), ("write_failure", "artifact_write_failed")],
)
def test_job_failures(
    make_evaluation_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch, mode: str, code: str
) -> None:
    evaluation_worker_mode(monkeypatch, mode)
    with make_evaluation_client(**({"job_timeout_seconds": 1} if mode == "timeout" else {})) as client:
        response = client.post(POST_URL, files=evaluation_parts())
        assert response.status_code == 202, response.text
        job = response.json()
        final = wait_status(client, job["links"]["self"], {"failed"})
        assert final["error"]["code"] == code
        assert "/private" not in json.dumps(final)
        assert_error(client.get(job["links"]["result"]), 409, "job_failed")
        assert {item["role"] for item in check_artifacts(client, job)} == {
            "input_original",
            "input_prepared",
            "request_config",
        }


def test_per_metric_failure_keeps_other_values(
    make_evaluation_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_worker_mode(monkeypatch, "metric_failure")
    with make_evaluation_client() as client:
        job = client.post(POST_URL, files=evaluation_parts()).json()
        result = completed_result(client, job)
        groups = result["samples"][0]["metrics"]["druglikeness"]
        assert groups["qed"]["status"] == "failed" and "value" not in groups["qed"]
        assert groups["sa_normalized"]["status"] == "succeeded"
        assert groups["validity"]["value"] is True
        check_artifacts(client, job)


def test_source_validation(make_evaluation_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch) -> None:
    block_workers(monkeypatch)
    with make_evaluation_client() as client:
        config = {
            "source": {"type": "inference_job", "job_id": str(uuid4()), "sample_ids": [0]},
            "metrics": ["druglikeness"],
            "docking": None,
        }
        assert_error(client.post(POST_URL, files=evaluation_parts(config)), 404, "job_not_found")
        inference = client.post("/api/v1/inference/jobs", files=parts()).json()
        config["source"]["job_id"] = inference["job_id"]
        assert_error(client.post(POST_URL, files=evaluation_parts(config)), 409, "source_job_not_succeeded")
        evaluation = client.post(POST_URL, files=evaluation_parts()).json()
        config["source"]["job_id"] = evaluation["job_id"]
        assert_error(client.post(POST_URL, files=evaluation_parts(config)), 422, "validation_error", "invalid_job_kind")
        for ids in ([], [0, 0], [-1], [True], ["0"]):
            config["source"]["sample_ids"] = ids
            assert_error(client.post(POST_URL, files=evaluation_parts(config)), 422, "validation_error")


def test_evaluation_openapi(make_evaluation_client: Callable[..., TestClient]) -> None:
    with make_evaluation_client() as client:
        schema = client.get("/openapi.json").json()
        operation = schema["paths"][POST_URL]["post"]
        config = operation["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]["config"]
        assert config["contentSchema"]["required"] == ["source", "metrics", "docking"]
        assert len(config["contentSchema"]["properties"]["source"]["oneOf"]) == 2
        assert "$ref" not in json.dumps(config["contentSchema"])
        result = schema["paths"]["/api/v1/jobs/{job_id}/result"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert len(result["oneOf"]) == 2


def invalid_ligand_sdf() -> bytes:
    molecule = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
    return (Chem.MolToMolBlock(molecule) + "$$$$\n").encode()


@pytest.mark.parametrize("with_druglikeness", [True, False])
def test_invalid_chemical_structure_is_a_persisted_evaluation_outcome(
    make_evaluation_client: Callable[..., TestClient], with_druglikeness: bool
) -> None:
    config = {
        "source": {"type": "upload"},
        "metrics": ["scaffold_preservation"] + (["druglikeness"] if with_druglikeness else []),
        "docking": None,
    }
    with make_evaluation_client() as client:
        response = client.post(POST_URL, files=evaluation_parts(config, ligand=invalid_ligand_sdf()))
        assert response.status_code == 202, response.text
        result = completed_result(client, response.json())
        groups = result["samples"][0]["metrics"]
        assert groups["scaffold_preservation"] == {"status": "skipped", "reason": "invalid_molecule"}
        if with_druglikeness:
            assert groups["druglikeness"]["validity"] == {"status": "succeeded", "value": False}
            assert groups["druglikeness"]["qed"]["status"] == "skipped"
            assert result["warnings"] == []
        else:
            assert result["warnings"][0]["code"] == "no_evaluation_values"


def test_atom_limits_apply_to_invalid_ligands(make_evaluation_client: Callable[..., TestClient]) -> None:
    with make_evaluation_client(max_num_ligand_atoms=5) as client:
        assert_error(
            client.post(POST_URL, files=evaluation_parts(ligand=invalid_ligand_sdf())),
            422,
            "validation_error",
            "limit_exceeded",
        )


def test_real_quickvina_submission_result_and_provenance(make_evaluation_client: Callable[..., TestClient]) -> None:
    if not shutil.which("obabel") or not os.access(PATH_QVINA2, os.X_OK):
        pytest.skip("Real docking requires Open Babel and QuickVina executables.")
    ligand = (ROOT / "examples/4m7t_fragment.sdf").read_bytes()
    config = {
        "source": {"type": "upload"},
        "metrics": ["druglikeness", "scaffold_preservation", "docking"],
        "docking": DOCKING,
    }
    with make_evaluation_client(disabled=frozenset({"inference"})) as client:
        response = client.post(POST_URL, files=evaluation_parts(config, ligand=ligand))
        assert response.status_code == 202, response.text
        job = response.json()
        result = completed_result(client, job)
        groups = result["samples"][0]["metrics"]
        assert groups["docking"]["status"] == "succeeded", result
        assert math.isfinite(groups["docking"]["value"]["affinity_kcal_mol"])
        assert groups["docking"]["value"]["num_poses"] >= 1
        assert groups["scaffold_preservation"]["value"]["contains_fragment"] is True
        manifest = check_artifacts(client, job)
        stored = next(
            item for item in manifest if item["role"] == "input_original" and item["filename"] == "ligand_sdf.sdf"
        )
        assert client.get(stored["url"]).content == ligand
        provenance = client.get(next(item["url"] for item in manifest if item["role"] == "provenance")).json()
        assert provenance["docking"]["seed"] == DOCKING["seed"]
        assert provenance["docking"]["receptor_ph"] == 7.4
        assert provenance["docking"]["external_command_timeout_seconds"] == 60
        assert len(provenance["docking"]["box_center_angstrom"]) == 3
        assert all(value >= 10 for value in provenance["docking"]["box_size_angstrom"])
        assert provenance["environment"]["dependencies"]["quickvina"] != "unavailable"


@pytest.mark.parametrize("field", ["fragment_sdf", "pocket_pdb", "reference_ligand_sdf"])
def test_metric_specific_files_are_required(make_evaluation_client: Callable[..., TestClient], field: str) -> None:
    config = {"source": {"type": "upload"}, "metrics": ["scaffold_preservation", "docking"], "docking": DOCKING}
    form = [item for item in evaluation_parts(config) if item[0] != field]
    with make_evaluation_client() as client:
        assert_error(client.post(POST_URL, files=form), 422, "validation_error", "missing_field")


def test_source_jobs_forbid_uploads(make_evaluation_client: Callable[..., TestClient]) -> None:
    config = {
        "source": {"type": "inference_job", "job_id": str(uuid4()), "sample_ids": [0]},
        "metrics": ["druglikeness"],
        "docking": None,
    }
    form = evaluation_parts(config) + [("ligand_sdf", ("ligand.sdf", molecule_sdf("CC"), "application/octet-stream"))]
    with make_evaluation_client() as client:
        assert_error(client.post(POST_URL, files=form), 422, "validation_error", "unexpected_field")


@pytest.mark.parametrize("field", ["fragment_sdf", "reference_ligand_sdf"])
def test_fragment_and_reference_must_be_connected(
    make_evaluation_client: Callable[..., TestClient], field: str
) -> None:
    config = {"source": {"type": "upload"}, "metrics": ["scaffold_preservation", "docking"], "docking": DOCKING}
    with make_evaluation_client() as client:
        assert_error(
            client.post(POST_URL, files=replace_part(evaluation_parts(config), field, molecule_sdf("CC.CC"))),
            422,
            "validation_error",
            "invalid_structure",
        )


def test_transport_validation_and_safe_filenames(
    make_evaluation_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    block_workers(monkeypatch)
    form = evaluation_parts()
    with make_evaluation_client() as client:
        assert_error(client.post(POST_URL, json=EVALUATION_CONFIG_EXAMPLE), 415, "unsupported_media_type")
        assert_error(
            client.post(POST_URL, files=replace_part(form, "ligand_sdf", molecule_sdf("C"), "ligand.exe")),
            415,
            "unsupported_media_type",
        )
        request = httpx2.Request("POST", "http://testserver" + POST_URL, files=form)
        body = request.read()
        headers = dict(request.headers)
        headers.pop("content-length")
        assert_error(client.post(POST_URL, content=body[:-20], headers=headers), 400, "malformed_request")
        response = client.post(POST_URL, files=replace_part(form, "ligand_sdf", molecule_sdf("CC"), "../../ligand.SDF"))
        assert response.status_code == 202, response.text
        job = response.json()
        running = wait_status(client, job["links"]["self"], {"running"})
        assert running["phase"] == "evaluating" and running["progress"] is None
        for artifact in check_artifacts(client, job):
            assert "/" not in artifact["filename"] and "\\" not in artifact["filename"]
