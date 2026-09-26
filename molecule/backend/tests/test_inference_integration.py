"""Real multipart, RDKit/BioPython preparation, persistence, worker processes, and HTTP downloads."""

import copy
import hashlib
import io
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid4

import httpx2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

import ace_backend.app as api
import ace_backend.dispatcher as dispatch
from ace_backend.jobs_schema import CONFIG_EXAMPLE
from ace_backend.schemas import (
    AvailableFeature,
    Capabilities,
    EvaluationCapabilities,
    OperationalLimits,
    UnavailableFeature,
)
from ace_backend.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
FILES = {"pocket_pdb": "pocket.pdb", "fragment_sdf": "fragment.sdf", "reference_ligand_sdf": "ligand.sdf"}
POST_URL = "/api/v1/inference/jobs"
FileParts = list[tuple[str, tuple[str | None, bytes | str, str]]]


def parts(example: str = "4m7t", config: dict[str, object] | None = None) -> FileParts:
    return [
        (field, (suffix, (ROOT / "examples" / f"{example}_{suffix}").read_bytes(), "application/octet-stream"))
        for field, suffix in FILES.items()
    ] + [("config", (None, json.dumps(CONFIG_EXAMPLE if config is None else config), "application/json"))]


def replace_part(form: FileParts, field: str, content: bytes | str, filename: str | None = None) -> FileParts:
    return [
        (key, (filename if filename is not None else value[0], content, value[2])) if key == field else (key, value)
        for key, value in form
    ]


def molecule_sdf(smiles: str, *, is_3d: bool = True) -> bytes:
    molecule = Chem.MolFromSmiles(smiles)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(is_3d)
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(index, (index * 1.3, 0.0, 0.1 * index if is_3d else 0.0))
    molecule.AddConformer(conformer)
    return (Chem.MolToMolBlock(molecule) + "$$$$\n").encode()


def assert_error(response: httpx2.Response, status: int, code: str, detail: str | None = None) -> dict[str, object]:
    assert response.status_code == status, response.text
    payload = response.json()
    assert payload["request_id"] == response.headers["X-Request-ID"]
    UUID(payload["request_id"])
    assert response.headers["Cache-Control"] == "no-store"
    assert payload["error"]["code"] == code
    if detail is not None:
        assert payload["error"]["details"][0]["code"] == detail
    assert str(ROOT) not in response.text
    return payload


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch) -> Callable[..., TestClient]:
    def readiness(settings: Settings) -> Capabilities:
        return Capabilities(
            inference=AvailableFeature(),
            limits=settings.limits,
            evaluation=EvaluationCapabilities(
                druglikeness=AvailableFeature(), scaffold_preservation=AvailableFeature(), docking=AvailableFeature()
            ),
        )

    monkeypatch.setattr(api, "detect_capabilities", readiness)

    # A real process reserves the execution slot without using a GPU. Shutdown must kill it.
    def blocked_command(root: Path, device: str) -> tuple[str, ...]:
        return (sys.executable, "-c", "import time; time.sleep(600)")

    monkeypatch.setattr(dispatch, "worker_command", blocked_command)

    def factory(**limits: int) -> TestClient:
        return TestClient(api.create_app(Settings(limits=OperationalLimits(**limits))), raise_server_exceptions=False)

    return factory


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> Iterator[TestClient]:
    with make_client() as session:
        yield session


def worker_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    def command(root: Path, device: str) -> tuple[str, ...]:
        return (sys.executable, str(Path(__file__).with_name("fake_inference_worker.py")), str(root), device, mode)

    monkeypatch.setattr(dispatch, "worker_command", command)


def wait_status(client: TestClient, url: str, statuses: set[str], timeout: float = 30) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in statuses:
            return body
        time.sleep(0.03)
    pytest.fail(f"Job did not reach {statuses}: {body}")


def test_accepts_and_persists_inputs_with_request_headers(client: TestClient, tmp_path: Path) -> None:
    response = client.post(POST_URL, files=parts())
    assert response.status_code == 202, response.text
    job = response.json()
    assert set(job) == {"job_id", "kind", "status", "created_at", "updated_at", "links"}
    UUID(job["job_id"])
    assert job["status"] == "queued"
    assert job["kind"] == "inference"
    assert job["created_at"] == job["updated_at"]
    assert job["created_at"].endswith("Z")
    assert response.headers["Location"] == job["links"]["self"]
    assert response.headers["Retry-After"] == "2"
    assert response.headers["Cache-Control"] == "no-store"
    running = wait_status(client, job["links"]["self"], {"running"})
    assert running["phase"] == "loading_models" and running["progress"] is None
    assert "finished_at" not in running
    assert client.get(job["links"]["self"]).headers["Retry-After"] == "2"
    assert_error(client.get(job["links"]["result"]), 409, "result_not_ready")
    manifest = client.get(job["links"]["artifacts"]).json()["artifacts"]
    assert len(manifest) == 7
    for artifact in manifest:
        fetched = client.get(artifact["url"])
        assert fetched.status_code == 200
        assert fetched.headers["Content-Disposition"].startswith("inline;")
        assert hashlib.sha256(fetched.content).hexdigest() == artifact["sha256"]
        assert len(fetched.content) == artifact["size_bytes"]
    directory = tmp_path / "jobs" / job["job_id"]
    assert json.loads((directory / "request.json").read_text()) == CONFIG_EXAMPLE
    for field, suffix in FILES.items():
        original = directory / "original" / (field + Path(suffix).suffix)
        assert original.read_bytes() == (ROOT / "examples" / f"4m7t_{suffix}").read_bytes()
    preparation = json.loads((directory / "preparation.json").read_text())
    assert preparation["resolved_num_ligand_atoms"] >= preparation["fragment_atom_count"]
    assert preparation["pocket_selection"]["cutoff_angstrom"] == 8.0
    assert preparation["pocket_selection"]["residues"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("preset", "unapproved"),
        ("num_samples", "2"),
        ("num_samples", True),
        ("num_samples", 0),
        ("num_samples", 17),
        ("num_samples", 2.0),
        ("seed", -1),
        ("seed", 4294967296),
        ("num_sampling_steps", 9),
        ("num_sampling_steps", 2001),
        ("num_ligand_atoms", 0),
        ("num_ligand_atoms", "30"),
        ("num_ligand_atoms", True),
        ("ace", None),
        ("ace.omega", "1.4"),
        ("ace.omega", True),
        ("ace.omega", float("nan")),
        ("ace.omega", float("inf")),
        ("ace.omega", -0.1),
        ("ace.omega", 10.1),
        ("ace.diffusion_scale", 0),
        ("ace.b1", 101),
        ("ace.b2", -1),
        ("checkpoint", "/private/model.ckpt"),
        ("ace.unknown", 10),
        ("moe", {"omega": 1.4, "diffusion_scale": 2.0}),
    ],
)
def test_strict_config_rejects_invalid_or_extra_values(
    client: TestClient, field: str, value: object, tmp_path: Path
) -> None:
    config = copy.deepcopy(CONFIG_EXAMPLE)
    if field.startswith("ace."):
        config["ace"][field.split(".")[1]] = value
    else:
        config[field] = value
    payload = assert_error(client.post(POST_URL, files=parts(config=config)), 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == f"config.{field}"
    assert not list((tmp_path / "jobs").glob("*/job.json"))


@pytest.mark.parametrize(
    "field", ["preset", "num_samples", "seed", "num_sampling_steps", "num_ligand_atoms", "ace", "ace.b1"]
)
def test_scientific_fields_are_required(client: TestClient, field: str) -> None:
    config = copy.deepcopy(CONFIG_EXAMPLE)
    if "." in field:
        del config["ace"][field.split(".")[1]]
    else:
        del config[field]
    assert_error(client.post(POST_URL, files=parts(config=config)), 422, "validation_error")


@pytest.mark.parametrize("preset", ["nr_scaffold_v1", "fkc_scaffold_v1"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("omega", -0.1),
        ("omega", 10.1),
        ("omega", True),
        ("omega", "1.4"),
        ("omega", float("nan")),
        ("diffusion_scale", 0),
        ("diffusion_scale", 10.1),
        ("diffusion_scale", float("inf")),
        ("b1", 30.0),
        ("b2", 0.336),
    ],
)
def test_constant_presets_reject_invalid_and_unused_parameters(
    client: TestClient, tmp_path: Path, preset: str, field: str, value: object
) -> None:
    config = {key: value for key, value in CONFIG_EXAMPLE.items() if key != "ace"}
    config.update(preset=preset, moe={"omega": 1.4, "diffusion_scale": 2.0, field: value})
    payload = assert_error(client.post(POST_URL, files=parts(config=config)), 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == f"config.moe.{field}"
    assert not list((tmp_path / "jobs").glob("*/job.json"))


@pytest.mark.parametrize("preset", ["nr_scaffold_v1", "fkc_scaffold_v1"])
@pytest.mark.parametrize("field", ["omega", "diffusion_scale"])
def test_constant_parameters_are_required(client: TestClient, preset: str, field: str) -> None:
    config = {key: value for key, value in CONFIG_EXAMPLE.items() if key != "ace"}
    parameters = {"omega": 1.4, "diffusion_scale": 2.0}
    del parameters[field]
    config.update(preset=preset, moe=parameters)
    payload = assert_error(client.post(POST_URL, files=parts(config=config)), 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == f"config.moe.{field}"


@pytest.mark.parametrize("preset", ["nr_scaffold_v1", "fkc_scaffold_v1"])
def test_constant_presets_forbid_ace_parameters(client: TestClient, preset: str) -> None:
    config = {**CONFIG_EXAMPLE, "preset": preset}
    payload = assert_error(client.post(POST_URL, files=parts(config=config)), 422, "validation_error")
    assert {detail["field"] for detail in payload["error"]["details"]} == {"config.moe", "config.ace"}
    config["moe"] = {"omega": 1.4, "diffusion_scale": 2.0}
    payload = assert_error(client.post(POST_URL, files=parts(config=config)), 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "config.ace"


@pytest.mark.parametrize("field", [*FILES, "config"])
def test_missing_and_duplicate_parts(client: TestClient, field: str) -> None:
    form = parts()
    assert_error(
        client.post(POST_URL, files=[item for item in form if item[0] != field]),
        422,
        "validation_error",
        "missing_field",
    )
    assert_error(
        client.post(POST_URL, files=form + [next(item for item in form if item[0] == field)]),
        422,
        "validation_error",
        "duplicate_field",
    )


@pytest.mark.parametrize(
    ("raw", "status", "detail"),
    [
        ("{", 400, "invalid_json"),
        (b"\xff", 400, "invalid_json"),
        ('{"seed": 1, "seed": 2}', 422, "duplicate_field"),
    ],
)
def test_malformed_json(client: TestClient, raw: str | bytes, status: int, detail: str) -> None:
    assert_error(
        client.post(POST_URL, files=replace_part(parts(), "config", raw)),
        status,
        "malformed_request" if status == 400 else "validation_error",
        detail,
    )


def test_media_types_and_part_types(client: TestClient) -> None:
    assert_error(client.post(POST_URL, json=CONFIG_EXAMPLE), 415, "unsupported_media_type")
    assert_error(
        client.post(POST_URL, files=parts() + [("unknown", (None, "value", "text/plain"))]),
        422,
        "validation_error",
        "unknown_field",
    )
    assert_error(
        client.post(POST_URL, files=replace_part(parts(), "config", json.dumps(CONFIG_EXAMPLE), "config.json")),
        422,
        "validation_error",
        "invalid_type",
    )
    assert_error(
        client.post(POST_URL, files=replace_part(parts(), "fragment_sdf", molecule_sdf("CC"), "fragment.txt")),
        415,
        "unsupported_media_type",
    )
    form = [(key, (None, value[1], value[2])) if key == "pocket_pdb" else (key, value) for key, value in parts()]
    assert_error(client.post(POST_URL, files=form), 422, "validation_error", "invalid_type")


def test_malformed_and_truncated_multipart(client: TestClient) -> None:
    assert_error(
        client.post(POST_URL, content=b"bad", headers={"Content-Type": "multipart/form-data"}), 400, "malformed_request"
    )
    request = httpx2.Request("POST", "http://testserver" + POST_URL, files=parts())
    body = request.read()
    assert_error(
        client.post(POST_URL, content=body[:-12], headers={"Content-Type": request.headers["Content-Type"]}),
        400,
        "malformed_request",
    )


@pytest.mark.parametrize(
    ("limit", "value", "field"), [("max_file_bytes", 10, "pocket_pdb"), ("max_config_bytes", 10, "config")]
)
def test_part_size_limits(make_client: Callable[..., TestClient], limit: str, value: int, field: str) -> None:
    with make_client(**{limit: value}) as client:
        response = client.post(POST_URL, files=parts())
        assert assert_error(response, 413, "payload_too_large")["error"]["details"][0]["field"] == field


def test_total_limit_includes_overhead_and_chunked_body(make_client: Callable[..., TestClient]) -> None:
    request = httpx2.Request("POST", "http://testserver" + POST_URL, files=parts())
    content = request.read()
    maximum = len(content) - 1
    with make_client(max_request_bytes=maximum, max_file_bytes=maximum, max_config_bytes=min(maximum, 16384)) as client:
        assert_error(
            client.post(POST_URL, content=content, headers={"Content-Type": request.headers["Content-Type"]}),
            413,
            "payload_too_large",
        )
        assert_error(
            client.post(
                POST_URL,
                content=iter([content[:100], content[100:]]),
                headers={"Content-Type": request.headers["Content-Type"]},
            ),
            413,
            "payload_too_large",
        )


@pytest.mark.parametrize(
    ("limit", "value", "field"),
    [
        ("max_num_samples", 1, "config.num_samples"),
        ("max_num_sampling_steps", 10, "config.num_sampling_steps"),
        ("max_num_ligand_atoms", 1, "fragment_sdf"),
        ("max_pocket_atoms", 1, "pocket_pdb"),
    ],
)
def test_server_limits(make_client: Callable[..., TestClient], limit: str, value: int, field: str) -> None:
    with make_client(**{limit: value}) as client:
        payload = assert_error(client.post(POST_URL, files=parts()), 422, "validation_error", "limit_exceeded")
        assert payload["error"]["details"][0]["field"] == field


@pytest.mark.parametrize(
    ("content", "detail"),
    [
        (b"", "invalid_structure"),
        (b"not an SDF", "invalid_structure"),
        (molecule_sdf("CC") * 2, "multiple_records"),
        (molecule_sdf("CC") + b"invalid second record", "multiple_records"),
        (molecule_sdf("C.C"), "invalid_structure"),
        (molecule_sdf("CC", is_3d=False), "coordinates_required"),
        (molecule_sdf("CS"), "unsupported_fragment_atom"),
        (molecule_sdf("C*"), "unsupported_fragment_atom"),
    ],
)
def test_invalid_fragment_chemistry(client: TestClient, content: bytes, detail: str) -> None:
    assert_error(
        client.post(POST_URL, files=replace_part(parts(), "fragment_sdf", content)), 422, "validation_error", detail
    )


def test_multiple_pdb_models_and_nonfinite_coordinates(client: TestClient) -> None:
    pdb = (ROOT / "examples/4m7t_pocket.pdb").read_bytes()
    multiple = b"MODEL        1\n" + pdb + b"ENDMDL\nMODEL        2\n" + pdb + b"ENDMDL\n"
    assert_error(
        client.post(POST_URL, files=replace_part(parts(), "pocket_pdb", multiple)),
        422,
        "validation_error",
        "multiple_models",
    )
    lines = pdb.decode().splitlines(keepends=True)
    index = next(index for index, line in enumerate(lines) if line.startswith("ATOM  "))
    lines[index] = lines[index][:30] + "     nan" + lines[index][38:]
    assert_error(
        client.post(POST_URL, files=replace_part(parts(), "pocket_pdb", "".join(lines))),
        422,
        "validation_error",
        "coordinates_required",
    )


def test_atom_count_and_empty_pocket(client: TestClient) -> None:
    config = {**CONFIG_EXAMPLE, "num_ligand_atoms": 1}
    assert_error(client.post(POST_URL, files=parts(config=config)), 422, "validation_error", "atom_count_too_small")
    assert_error(
        client.post(POST_URL, files=replace_part(parts(), "reference_ligand_sdf", molecule_sdf("C"))),
        422,
        "validation_error",
        "atom_count_too_small",
    )
    reference = Chem.SDMolSupplier(str(ROOT / "examples/4m7t_ligand.sdf"), removeHs=False)[0]
    conformer = reference.GetConformer()
    for index in range(reference.GetNumAtoms()):
        conformer.SetAtomPosition(index, np.asarray(conformer.GetAtomPosition(index)) + 10000)
    far = (Chem.MolToMolBlock(reference) + "$$$$\n").encode()
    assert_error(
        client.post(POST_URL, files=replace_part(parts(), "reference_ligand_sdf", far)),
        422,
        "validation_error",
        "empty_pocket",
    )


def test_normalization_preserves_isotopes_charge_order_and_coordinates(client: TestClient) -> None:
    source = Chem.SDMolSupplier(str(ROOT / "examples/4m7t_fragment.sdf"), removeHs=False)[0]
    molecule = Chem.AddHs(source, addCoords=True)
    hydrogen = next(atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 1)
    hydrogen.SetIsotope(2)
    original = (Chem.MolToMolBlock(molecule) + "$$$$\n").encode()
    parsed = next(Chem.ForwardSDMolSupplier(io.BytesIO(original), removeHs=False))
    expected = Chem.RemoveHs(parsed)
    response = client.post(POST_URL, files=replace_part(parts(), "fragment_sdf", original))
    assert response.status_code == 202, response.text
    manifest = client.get(response.json()["links"]["artifacts"]).json()["artifacts"]
    artifact = next(
        item for item in manifest if item["role"] == "input_prepared" and item["filename"] == "fragment_sdf.sdf"
    )
    actual = next(Chem.ForwardSDMolSupplier(io.BytesIO(client.get(artifact["url"]).content), removeHs=False))
    assert [(atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge()) for atom in actual.GetAtoms()] == [
        (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge()) for atom in expected.GetAtoms()
    ]
    np.testing.assert_allclose(
        actual.GetConformer().GetPositions(), expected.GetConformer().GetPositions(), atol=0.0001
    )


def test_filenames_are_never_paths_and_repeated_posts_are_distinct(client: TestClient, tmp_path: Path) -> None:
    form = replace_part(
        parts(), "fragment_sdf", (ROOT / "examples/4m7t_fragment.sdf").read_bytes(), "../../escaped.sdf"
    )
    first, second = client.post(POST_URL, files=form), client.post(POST_URL, files=form)
    assert first.status_code == second.status_code == 202
    assert first.json()["job_id"] != second.json()["job_id"]
    assert not (tmp_path / "escaped.sdf").exists()
    for artifact in client.get(first.json()["links"]["artifacts"]).json()["artifacts"]:
        assert "/" not in artifact["filename"] and "\\" not in artifact["filename"]


def test_queue_capacity_is_atomic(make_client: Callable[..., TestClient], tmp_path: Path) -> None:
    with make_client(max_pending_jobs=1) as client:
        first = client.post(POST_URL, files=parts()).json()
        wait_status(client, first["links"]["self"], {"running"})
        with ThreadPoolExecutor(max_workers=3) as executor:
            responses = list(executor.map(lambda _: client.post(POST_URL, files=parts()), range(3)))
        assert sorted(response.status_code for response in responses) == [202, 429, 429]
        for response in responses:
            if response.status_code == 429:
                assert_error(response, 429, "queue_full")
                assert response.headers["Retry-After"] == "2"
        assert len(list((tmp_path / "jobs").glob("*/job.json"))) == 2
        assert not list((tmp_path / "jobs").glob(".submission-*"))


def test_zero_pending_capacity_still_accepts_an_idle_worker(make_client: Callable[..., TestClient]) -> None:
    with make_client(max_pending_jobs=0) as client:
        assert client.post(POST_URL, files=parts()).status_code == 202
        assert_error(client.post(POST_URL, files=parts()), 429, "queue_full")


def test_unavailable_inference_rejects_without_creating_job(
    make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = api.detect_capabilities
    monkeypatch.setattr(
        api,
        "detect_capabilities",
        lambda settings: original(settings).model_copy(
            update={"inference": UnavailableFeature(reason="cuda_unavailable")}
        ),
    )
    with make_client() as client:
        assert_error(client.post(POST_URL, files=parts()), 503, "inference_unavailable")
        assert client.get("/api/v1/capabilities").status_code == 200
    assert not list((tmp_path / "jobs").glob("*/job.json"))


def test_restart_fails_interrupted_jobs_and_releases_storage(make_client: Callable[..., TestClient]) -> None:
    with make_client() as client:
        running = client.post(POST_URL, files=parts()).json()
        wait_status(client, running["links"]["self"], {"running"})
        queued = client.post(POST_URL, files=parts()).json()
    with make_client() as client:
        for original in (running, queued):
            job = client.get(original["links"]["self"]).json()
            assert job["status"] == "failed" and job["error"]["code"] == "service_restarted"
            assert (job["started_at"] is None) == (original is queued)
            assert_error(client.get(original["links"]["result"]), 409, "job_failed")
        assert client.post(POST_URL, files=parts()).status_code == 202


@pytest.mark.parametrize("example", ["4m7t", "3nfb", "4yhj"])
@pytest.mark.parametrize(
    ("filename", "sampler_name", "use_logq", "do_resample"),
    [
        ("inference-config-nr.json", "NRSampler", False, False),
        ("inference-config-fkc.json", "FKCSampler", False, True),
        ("inference-config.json", "ACESampler", True, True),
    ],
)
def test_submission_to_results_and_downloads(
    make_client: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
    example: str,
    filename: str,
    sampler_name: str,
    use_logq: bool,
    do_resample: bool,
) -> None:
    config = json.loads((ROOT / "examples" / filename).read_text())
    config["num_samples"] = 2
    worker_mode(monkeypatch, "success")
    with make_client() as client:
        submitted = client.post(POST_URL, files=parts(example, config=config))
        assert submitted.status_code == 202, submitted.text
        job = submitted.json()
        final = wait_status(client, job["links"]["self"], {"succeeded", "failed"})
        assert final["status"] == "succeeded", final
        assert "phase" not in final and "progress" not in final and "error" not in final
        assert "Retry-After" not in client.get(job["links"]["self"]).headers
        result_response = client.get(job["links"]["result"])
        assert result_response.status_code == 200
        result = result_response.json()
        assert result["summary"] == {"requested": 2, "available": 1, "invalid": 1}
        assert [sample["sample_id"] for sample in result["samples"]] == [0, 1]
        assert set(result["samples"][1]) == {"sample_id", "status", "error"}
        manifest = client.get(job["links"]["artifacts"]).json()["artifacts"]
        references = list(result["inputs"].values()) + [result["samples"][0]["sdf"]]
        artifacts = {artifact["artifact_id"]: artifact for artifact in manifest}
        for reference in references:
            assert artifacts[reference["artifact_id"]]["url"] == reference["url"]
        for artifact in manifest:
            content = client.get(artifact["url"] + "?download=true")
            assert content.status_code == 200
            assert content.headers["Content-Disposition"].startswith("attachment;")
            assert content.headers["Cache-Control"] == "no-store"
            assert len(content.content) == artifact["size_bytes"]
            assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
        generated = client.get(result["samples"][0]["sdf"]["url"]).content
        molecule = next(Chem.ForwardSDMolSupplier(io.BytesIO(generated), removeHs=False))
        normalized = next(
            Chem.ForwardSDMolSupplier(
                io.BytesIO(client.get(result["inputs"]["reference_ligand_sdf"]["url"]).content), removeHs=False
            )
        )
        np.testing.assert_allclose(
            molecule.GetConformer().GetPositions(), normalized.GetConformer().GetPositions(), atol=0.0001
        )
        assert molecule.GetNumAtoms() == result["samples"][0]["atom_count"]
        resolved = client.get(
            next(artifact["url"] for artifact in manifest if artifact["role"] == "resolved_config")
        ).json()
        assert resolved["data"]["num_ligand_atoms"] == result["resolved_num_ligand_atoms"]
        assert resolved["preset"] == config["preset"]
        assert resolved["sampler"]["seed"] == config["seed"]
        assert resolved["sampler"]["name"] == sampler_name
        assert resolved["sampler"]["use_logq"] is use_logq
        assert resolved["sampler"]["do_resample"] is do_resample
        weights = resolved["moe"]["exponents"]
        assert [entry["weight_fn"]["name"] for entry in weights.values()] == [
            "ConstantWeight",
            "ConstantWeight",
            "ConstantWeight",
            "ACEBumpWeight" if sampler_name == "ACESampler" else "ConstantWeight",
        ]
        if sampler_name != "ACESampler":
            assert all(set(entry["weight_fn"]) == {"name", "omega"} for entry in weights.values())
        provenance = client.get(
            next(artifact["url"] for artifact in manifest if artifact["role"] == "provenance")
        ).json()
        assert provenance["code"]["repositories"][0]["commit"]
        assert provenance["environment"]["validation"] == "substituted models; no CUDA inference"
        assert provenance["request"] == config
        request_artifact = next(artifact for artifact in manifest if artifact["role"] == "request_config")
        assert client.get(request_artifact["url"]).json() == config
        assert_error(
            client.get(f"/api/v1/jobs/{uuid4()}/artifacts/{result['samples'][0]['sdf']['artifact_id']}"),
            404,
            "job_not_found",
        )
    with make_client() as restarted:
        assert restarted.get(job["links"]["self"]).json() == final
        assert restarted.get(job["links"]["result"]).json() == result
        assert restarted.get(result["samples"][0]["sdf"]["url"]).content == generated


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("load_failure", "model_load_failed"),
        ("sampling_failure", "inference_failed"),
        ("oom", "gpu_out_of_memory"),
        ("write_failure", "artifact_write_failed"),
    ],
)
def test_worker_failures_are_job_failures(
    make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch, mode: str, error: str
) -> None:
    worker_mode(monkeypatch, mode)
    with make_client() as client:
        job = client.post(POST_URL, files=parts()).json()
        final = wait_status(client, job["links"]["self"], {"failed", "succeeded"})
        assert final["status"] == "failed", final
        assert final["error"]["code"] == error
        assert "private" not in json.dumps(final) and "/server/" not in json.dumps(final)
        assert "phase" not in final and "progress" not in final
        assert_error(client.get(job["links"]["result"]), 409, "job_failed")
        assert len(client.get(job["links"]["artifacts"]).json()["artifacts"]) == 7


def test_all_invalid_samples_are_a_completed_result(
    make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_mode(monkeypatch, "all_invalid")
    with make_client() as client:
        job = client.post(POST_URL, files=parts()).json()
        assert wait_status(client, job["links"]["self"], {"succeeded", "failed"})["status"] == "succeeded"
        result = client.get(job["links"]["result"]).json()
        assert result["summary"] == {"requested": 2, "available": 0, "invalid": 2}
        assert result["warnings"][0]["code"] == "no_valid_samples"
        assert all(set(sample) == {"sample_id", "status", "error"} for sample in result["samples"])


def test_timeout_terminates_process_and_queue_continues(
    make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def command(root: Path, device: str) -> tuple[str, ...]:
        return (
            sys.executable,
            "-c",
            "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(600)",
            str(root / "pid"),
        )

    monkeypatch.setattr(dispatch, "worker_command", command)
    with make_client(job_timeout_seconds=1) as client:
        jobs = [client.post(POST_URL, files=parts()).json() for _ in range(2)]
        for job in jobs:
            final = wait_status(client, job["links"]["self"], {"failed"}, timeout=8)
            assert final["error"]["code"] == "job_timeout"
            pid = int((tmp_path / "jobs" / job["job_id"] / "pid").read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)


def test_unknown_job_artifact_and_query_validation(client: TestClient) -> None:
    assert_error(client.get("/api/v1/jobs/not-a-uuid"), 404, "job_not_found")
    job = client.post(POST_URL, files=parts()).json()
    assert_error(client.get(job["links"]["artifacts"] + f"/{uuid4()}"), 404, "artifact_not_found")
    artifacts = client.get(job["links"]["artifacts"]).json()["artifacts"]
    assert_error(client.get(artifacts[0]["url"] + "?download=invalid"), 422, "validation_error")


def test_openapi_describes_config_and_cors_supports_submission(
    make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACE_API_CORS_ORIGINS", '["http://localhost:5173"]')
    with make_client() as client:
        schema = client.get("/openapi.json").json()["paths"][POST_URL]["post"]["requestBody"]["content"][
            "multipart/form-data"
        ]["schema"]
        config = schema["properties"]["config"]
        assert config["type"] == "string"
        assert json.loads(config["example"]) == CONFIG_EXAMPLE
        branches = config["contentSchema"]["oneOf"]
        by_preset = {branch["properties"]["preset"]["const"]: branch for branch in branches}
        assert set(by_preset) == {"nr_scaffold_v1", "fkc_scaffold_v1", "ace_scaffold_v1"}
        for preset, branch in by_preset.items():
            parameter_key = "ace" if preset == "ace_scaffold_v1" else "moe"
            assert set(branch["required"]) == {
                "preset",
                "num_samples",
                "seed",
                "num_sampling_steps",
                "num_ligand_atoms",
                parameter_key,
            }
            assert branch["additionalProperties"] is False
            parameters = branch["properties"][parameter_key]
            assert parameters["additionalProperties"] is False
            assert set(parameters["required"]) == (
                {"omega", "diffusion_scale", "b1", "b2"} if parameter_key == "ace" else {"omega", "diffusion_scale"}
            )
        assert "$ref" not in json.dumps(config["contentSchema"])
        response = client.options(
            POST_URL,
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert response.status_code == 200


def test_submission_storage_failure_cleans_up_inputs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = Path.write_bytes

    def fail_prepared_write(path: Path, data: bytes) -> int:
        if path.parent.name == "prepared":
            raise OSError("private /server/storage/path")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_prepared_write)
    response = client.post(POST_URL, files=parts())
    assert_error(response, 500, "internal_error")
    assert "private" not in response.text
    assert not list((tmp_path / "jobs").glob("*/job.json"))
    assert not list((tmp_path / "jobs").glob(".submission-*"))


def test_storage_has_one_owner(make_client: Callable[..., TestClient]) -> None:
    with make_client():
        with pytest.raises(RuntimeError, match="already in use"):
            with make_client():
                pass


def test_worker_crash_releases_the_execution_slot(
    make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatch, "worker_command", lambda root, device: (sys.executable, "-c", "raise SystemExit(3)"))
    with make_client(max_pending_jobs=0) as client:
        for _ in range(2):
            response = client.post(POST_URL, files=parts())
            assert response.status_code == 202
            job = wait_status(client, response.json()["links"]["self"], {"failed"})
            assert job["error"]["code"] == "inference_failed"


def test_operator_removed_job_returns_not_found(
    make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dispatch, "worker_command", lambda root, device: (sys.executable, "-c", "raise SystemExit(3)"))
    with make_client() as client:
        job = client.post(POST_URL, files=parts()).json()
        wait_status(client, job["links"]["self"], {"failed"})
        shutil.rmtree(tmp_path / "jobs" / job["job_id"])
        for url in job["links"].values():
            assert_error(client.get(url), 404, "job_not_found")


def test_no_gpu_default_worker_reports_model_load_failure(
    make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise the production entry point and Linux parent-death setup without loading models onto a GPU.
    monkeypatch.setattr(
        dispatch,
        "worker_command",
        lambda root, device: (
            sys.executable,
            "-m",
            "ace_backend.worker",
            str(root),
            device,
            str(os.getpid()),
        ),
    )
    with make_client() as client:
        response = client.post(POST_URL, files=parts())
        assert response.status_code == 202
        job = wait_status(client, response.json()["links"]["self"], {"failed", "succeeded"})
        assert job["status"] == "failed"
        assert job["error"]["code"] == "model_load_failed"


def test_normalized_reference_keeps_precision_at_pocket_cutoff(client: TestClient) -> None:
    molecule = Chem.MolFromSmiles("C")
    conformer = Chem.Conformer(1)
    conformer.Set3D(True)
    conformer.SetAtomPosition(0, (0.00004, 0.0, 0.0))
    molecule.AddConformer(conformer)
    parameters = Chem.MolWriterParams()
    parameters.forceV3000 = True
    parameters.precision = 17
    sdf = (Chem.MolToMolBlock(molecule, parameters) + "$$$$\n").encode()
    pdb = b"ATOM      1  CA  ALA A   1       8.000   0.000   0.000  1.00  0.00           C\nEND\n"
    form = replace_part(parts(), "pocket_pdb", pdb)
    form = replace_part(form, "fragment_sdf", sdf)
    form = replace_part(form, "reference_ligand_sdf", sdf)
    response = client.post(POST_URL, files=form)
    assert response.status_code == 202, response.text
    artifacts = client.get(response.json()["links"]["artifacts"]).json()["artifacts"]
    prepared = next(
        artifact
        for artifact in artifacts
        if artifact["role"] == "input_prepared" and artifact["filename"] == "reference_ligand_sdf.sdf"
    )
    reread = next(Chem.ForwardSDMolSupplier(io.BytesIO(client.get(prepared["url"]).content), removeHs=False))
    np.testing.assert_array_equal(reread.GetConformer().GetPositions(), conformer.GetPositions())
    assert np.linalg.norm(np.array([8.0, 0, 0]) - reread.GetConformer().GetPositions()[0]) < 8
