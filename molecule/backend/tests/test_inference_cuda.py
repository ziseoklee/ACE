"""Opt-in real-model integration. Run separately from the substituted-model suite."""

import copy
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBIO, PDBParser
from fastapi.testclient import TestClient
from rdkit import Chem

from ace_backend.app import create_app
from ace_backend.jobs_schema import CONFIG_EXAMPLE
from ace_backend.molecule_io import serialize_sdf
from ace_backend.schemas import OperationalLimits
from ace_backend.settings import Settings

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not os.environ.get("ACE_TEST_CUDA_DEVICE"), reason="Set ACE_TEST_CUDA_DEVICE to opt in to real ACE inference."
    ),
]


def test_real_inference_preserves_the_uploaded_coordinate_frame() -> None:
    root = Path(__file__).resolve().parents[2]
    offset = np.array([100.0, -70.0, 150.0])
    pocket = PDBParser(QUIET=True).get_structure("pocket", str(root / "examples/4m7t_pocket.pdb"))
    for atom in pocket.get_atoms():
        atom.set_coord(atom.get_coord() + offset)
    pdb_buffer = io.StringIO()
    writer = PDBIO()
    writer.set_structure(pocket)
    writer.save(pdb_buffer)
    form = [("pocket_pdb", ("pocket.pdb", pdb_buffer.getvalue(), "chemical/x-pdb"))]
    reference_center = None
    for field, name in (("fragment_sdf", "fragment"), ("reference_ligand_sdf", "ligand")):
        molecule = Chem.SDMolSupplier(str(root / f"examples/4m7t_{name}.sdf"), removeHs=False)[0]
        conformer = molecule.GetConformer()
        for index in range(molecule.GetNumAtoms()):
            conformer.SetAtomPosition(index, np.array(conformer.GetAtomPosition(index)) + offset)
        if name == "ligand":
            reference_center = conformer.GetPositions().mean(axis=0)
        form.append((field, (f"{name}.sdf", serialize_sdf(molecule), "chemical/x-mdl-sdfile")))
    config = copy.deepcopy(CONFIG_EXAMPLE)
    config.update(num_samples=1, num_sampling_steps=100)
    form.append(("config", (None, json.dumps(config), "application/json")))
    settings = Settings(device=os.environ["ACE_TEST_CUDA_DEVICE"], limits=OperationalLimits(job_timeout_seconds=300))
    with TestClient(create_app(settings)) as client:
        capability = client.get("/api/v1/capabilities").json()
        assert capability["inference"]["available"], capability["inference"]
        response = client.post("/api/v1/inference/jobs", files=form)
        assert response.status_code == 202, response.text
        links = response.json()["links"]
        deadline = time.monotonic() + 310
        while time.monotonic() < deadline:
            job = client.get(links["self"]).json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.2)
        assert job["status"] == "succeeded", job
        result = client.get(links["result"]).json()
        assert result["summary"]["requested"] == 1
        artifacts = client.get(links["artifacts"]).json()["artifacts"]
        xyz = next(artifact for artifact in artifacts if artifact["filename"] == "sample_0.xyz")
        points = Chem.MolFromXYZBlock(client.get(xyz["url"]).text).GetConformer().GetPositions()
        assert np.isfinite(points).all()
        # A large common translation exposes output left in the model's centered frame.
        assert np.linalg.norm(points.mean(axis=0) - reference_center) < 25
        for sample in result["samples"]:
            if sample["status"] == "available":
                downloaded = client.get(sample["sdf"]["url"])
                molecule = next(Chem.ForwardSDMolSupplier(io.BytesIO(downloaded.content), removeHs=False))
                np.testing.assert_allclose(molecule.GetConformer().GetPositions(), points, atol=0.0001)
                metadata = next(
                    artifact for artifact in artifacts if artifact["artifact_id"] == sample["sdf"]["artifact_id"]
                )
                assert hashlib.sha256(downloaded.content).hexdigest() == metadata["sha256"]
        provenance = client.get(
            next(artifact["url"] for artifact in artifacts if artifact["role"] == "provenance")
        ).json()
        assert len(provenance["checkpoints"]) == 5
        assert provenance["environment"]["sampler_device"] == settings.device
