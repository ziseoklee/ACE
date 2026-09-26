"""Scientific adapter boundaries that do not require GPUs or model weights."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from rdkit import Chem

from ace_backend.inference_runtime import build_preset
from ace_backend.jobs_schema import CONFIG_EXAMPLE, INFERENCE_CONFIG_ADAPTER
from ace_backend.worker import publish_sample


class ZeroScheduler:
    def drift_coeff(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_ace_log_weight_derivative_handles_one_or_more_particles(batch_size: int) -> None:
    from sampling.probability_path import MoEProbabilityPath, PaddedProbabilityPath, ProbabilityPath

    scheduler = ZeroScheduler()
    base = ProbabilityPath(scheduler, lambda t, x: torch.zeros_like(x), reverse=False)
    padded = PaddedProbabilityPath(
        [base, base], [torch.tensor([True, True, False]), torch.tensor([False, False, True])]
    )
    path = MoEProbabilityPath(
        scheduler,
        [padded, padded],
        [torch.ones(3, dtype=torch.bool)] * 2,
        [lambda t: t.square(), lambda t: 3 * t],
        sample_size=3,
        node_feature_dim=3,
    )
    t = torch.linspace(0.2, 0.8, batch_size).unsqueeze(1)
    logq = torch.stack([torch.arange(1, batch_size + 1), torch.full((batch_size,), 2)], dim=1).unsqueeze(2).float()
    actual = path.get_dlog_weight(t, torch.zeros(batch_size, 3), use_logq=True, logq_tensor=logq)
    expected = 2 * t[:, 0] * logq[:, 0, 0] + 3 * logq[:, 1, 0]
    assert actual.shape == (batch_size,)
    torch.testing.assert_close(actual, expected)


def test_preset_matches_existing_four_expert_configuration() -> None:
    from hydra import compose, initialize_config_dir

    from configs import config as registry

    request = INFERENCE_CONFIG_ADAPTER.validate_python(CONFIG_EXAMPLE)
    sampler, moe = build_preset(request, "cuda:3")
    with initialize_config_dir(config_dir=str(Path(registry.__file__).parent), version_base=None):
        existing = compose(
            config_name="inference",
            overrides=[
                "moe.omega=1.4",
                "sampler.batch_size=2",
                "sampler.seed=42",
                "sampler.num_sampling_steps=500",
                "sampler.device=cuda:3",
            ],
        )
    assert OmegaConf.to_container(OmegaConf.structured(sampler), resolve=True) == OmegaConf.to_container(
        existing.sampler, resolve=True
    )
    assert OmegaConf.to_container(OmegaConf.structured(moe), resolve=True) == OmegaConf.to_container(
        existing.moe, resolve=True
    )


@pytest.mark.parametrize(
    ("filename", "name", "use_logq", "do_resample"),
    [
        ("inference-config-nr.json", "NRSampler", False, False),
        ("inference-config-fkc.json", "FKCSampler", False, True),
        ("inference-config.json", "ACESampler", True, True),
    ],
)
def test_presets_preserve_expert_exponents_and_explicit_settings(
    filename: str, name: str, use_logq: bool, do_resample: bool
) -> None:
    from inference.sampling_runtime import build_exponent_list

    raw = json.loads((Path(__file__).resolve().parents[2] / "examples" / filename).read_text())
    raw.update(num_samples=3, seed=123, num_sampling_steps=120)
    parameters = raw["ace"] if name == "ACESampler" else raw["moe"]
    parameters.update(omega=2.3, diffusion_scale=1.25)
    if name == "ACESampler":
        parameters.update(b1=7.0, b2=0.5)
    sampler, moe = build_preset(INFERENCE_CONFIG_ADAPTER.validate_python(raw), "cuda:2")
    assert (sampler.name, sampler.use_logq, sampler.do_resample) == (name, use_logq, do_resample)
    assert (sampler.batch_size, sampler.seed, sampler.num_sampling_steps, sampler.device) == (3, 123, 120, "cuda:2")
    assert moe.diffusion_scale == 1.25
    assert moe.global_scheduler_key == "GEODIFF"
    assert list(moe.components) == ["edm_fragment", "edm_ligand", "geodiff_fragment", "diffsbdd"]
    exponents = build_exponent_list(tuple(moe.components.items()), moe.exponents)
    t = torch.tensor([[0.0], [0.25], [0.5], [1.0]])
    actual = torch.cat([exponent(t) for exponent in exponents], dim=1)
    expected = torch.tensor([[-2.3, -1.3, 2.3, 2.3]]).expand(4, -1).clone()
    if name == "ACESampler":
        expected[:, 3] += (7.0 * t * (1 - t) + 0.5 * t)[:, 0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("problem", ["missing_conformer", "2d", "nonfinite", "invalid_valence"])
def test_invalid_generated_molecules_have_no_sdf(tmp_path: Path, problem: str) -> None:
    molecule = Chem.MolFromSmiles("CC")
    if problem == "invalid_valence":
        molecule = Chem.MolFromSmiles("C(F)(F)(F)(F)F", sanitize=False)
    if problem != "missing_conformer":
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        conformer.Set3D(problem != "2d")
        if problem == "nonfinite":
            conformer.SetAtomPosition(0, (float("nan"), 0, 0))
        molecule.AddConformer(conformer)
    (tmp_path / "work").mkdir()
    sample, artifact = publish_sample(tmp_path, "job", 3, molecule)
    assert sample.status == "invalid" and sample.error.code == "invalid_generated_molecule"
    assert sample.sample_id == 3 and artifact is None
    assert not list((tmp_path / "work").iterdir())


def test_generated_disconnected_molecule_preserves_hydrogens_and_coordinates(tmp_path: Path) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("C.C"))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(index, (10 + index, -2.5, 4.9))
    molecule.AddConformer(conformer)
    (tmp_path / "work").mkdir()
    sample, artifact = publish_sample(tmp_path, "job", 5, molecule)
    assert sample.status == "available" and sample.component_count == 2
    reread = next(
        Chem.ForwardSDMolSupplier(io.BytesIO((tmp_path / artifact.relative_path).read_bytes()), removeHs=False)
    )
    assert reread.GetNumAtoms() == molecule.GetNumAtoms() == sample.atom_count
    np.testing.assert_allclose(
        reread.GetConformer().GetPositions(), molecule.GetConformer().GetPositions(), atol=0.0001
    )
