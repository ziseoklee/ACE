"""Scientific report semantics: actual zero, failed calculations, topology, and docking."""

import math
import subprocess
from pathlib import Path
from typing import Never

import pytest
from rdkit import Chem

from ace_backend.evaluation_runtime import evaluate_molecule, has_evaluation_values
from ace_backend.evaluation_schema import DockingConfig, EvaluationConfig, MetricGroups, MetricName, UploadSource
from evaluation.backends import qvina
from evaluation.metrics import druglikeness


def config(*metrics: MetricName) -> EvaluationConfig:
    return EvaluationConfig(
        source=UploadSource(type="upload"),
        metrics=metrics,
        docking=DockingConfig(seed=123, exhaustiveness=3, num_modes=4, padding_angstrom=6.0)
        if "docking" in metrics
        else None,
    )


def evaluate(smiles: str, request: EvaluationConfig, *, fragment: str = "CC") -> MetricGroups:
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    before = Chem.MolToMolBlock(molecule, kekulize=False)
    result = evaluate_molecule(
        molecule,
        request,
        fragment=Chem.MolFromSmiles(fragment),
        pocket=Path("pocket.pdb"),
        reference=Chem.MolFromSmiles("CCO"),
    )
    assert Chem.MolToMolBlock(molecule, kekulize=False) == before
    return result


def test_api_values_match_legacy_metrics_on_valid_input() -> None:
    legacy = druglikeness.evaluate_druglikeness(Chem.MolFromSmiles("CCOc1ccccc1"))
    actual = evaluate("CCOc1ccccc1", config("druglikeness")).druglikeness
    assert actual.validity.value is True
    for field, old in (("qed", "QED"), ("sa_normalized", "SA"), ("logp", "LogP"), ("lipinski_legacy", "Lipinski")):
        assert getattr(actual, field).value == pytest.approx(legacy[old])


def test_zero_is_success_and_exception_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(druglikeness.QED, "qed", lambda mol: 0.0)
    assert evaluate("CCC", config("druglikeness")).druglikeness.qed.model_dump() == {
        "status": "succeeded",
        "value": 0.0,
    }

    def fail(mol: Chem.Mol) -> float:
        raise ValueError("private scientific details")

    monkeypatch.setattr(druglikeness.QED, "qed", fail)
    report = evaluate("CCC", config("druglikeness")).druglikeness.qed
    assert report.status == "failed" and report.error.code == "metric_failed"
    assert "value" not in report.model_dump()
    assert druglikeness.evaluate_druglikeness(Chem.MolFromSmiles("CCC"))["QED"] == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_metrics_fail_without_clamping_to_valid_values(monkeypatch: pytest.MonkeyPatch, bad: float) -> None:
    monkeypatch.setattr(druglikeness.QED, "qed", lambda mol: bad)
    monkeypatch.setattr(druglikeness.sascorer, "calculateScore", lambda mol: bad)
    reports = evaluate("CCC", config("druglikeness")).druglikeness
    assert reports.qed.status == reports.sa_normalized.status == "failed"
    assert reports.validity.value is True
    assert math.isfinite(reports.logp.value)


def test_logp_failure_prevents_legacy_score_but_keeps_other_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(mol: Chem.Mol) -> Never:
        raise RuntimeError("cannot calculate logp")

    monkeypatch.setattr(druglikeness.rdMolDescriptors, "CalcCrippenDescriptors", fail)
    reports = evaluate("CCC", config("druglikeness")).druglikeness
    assert reports.logp.status == reports.lipinski_legacy.status == "failed"
    assert reports.sa_normalized.status == "succeeded"


def test_unavailable_sa_is_not_a_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(druglikeness, "sascorer", None)
    actual = evaluate("CCC", config("druglikeness")).druglikeness.sa_normalized
    assert actual.status == "failed" and actual.error.code == "metric_unavailable"
    assert druglikeness.evaluate_druglikeness(Chem.MolFromSmiles("CCC"))["SA"] == 0.0


@pytest.mark.parametrize(
    ("ligand", "fragment", "expected"),
    [("CCCl", "CCl", True), ("CCC", "CCl", False), ("F[C@@H](Cl)Br", "F[C@H](Cl)Br", True)],
)
def test_topology_matching_ignores_chirality_and_inference_element_vocabulary(
    ligand: str, fragment: str, expected: bool
) -> None:
    result = evaluate(ligand, config("scaffold_preservation"), fragment=fragment)
    assert result.scaffold_preservation.value.contains_fragment is expected
    assert set(result.model_dump()) == {"scaffold_preservation"}


def test_invalid_molecule_skips_requested_metrics_without_running_docking(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> Never:
        pytest.fail("Docking must not run for invalid molecules")

    monkeypatch.setattr(qvina, "qvina_score_from_mol", forbidden)
    result = evaluate("C(C)(C)(C)(C)C", config("druglikeness", "scaffold_preservation", "docking"))
    assert result.druglikeness.validity.model_dump() == {"status": "succeeded", "value": False}
    assert result.druglikeness.qed.status == "skipped"
    assert result.scaffold_preservation.status == result.docking.status == "skipped"
    assert has_evaluation_values(result)
    assert not has_evaluation_values(evaluate("C(C)(C)(C)(C)C", config("scaffold_preservation")))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("timeout", "docking_timeout"),
        ("failure", "docking_failed"),
        ("empty", "docking_failed"),
        ("nonfinite", "docking_failed"),
        ("success", None),
    ],
)
def test_docking_passes_explicit_settings_and_preserves_failures(
    monkeypatch: pytest.MonkeyPatch, mode: str, expected: str | None
) -> None:
    request = config("docking", "druglikeness")

    def dock(protein: str, molecule: Chem.Mol, **kwargs: object) -> qvina.QvinaResult:
        assert protein == "pocket.pdb"
        assert (
            kwargs["seed"] == 123
            and kwargs["exhaustiveness"] == 3
            and kwargs["num_modes"] == 4
            and kwargs["pad"] == 6.0
        )
        assert Chem.MolToSmiles(kwargs["ref_mol"]) == "CCO"
        if mode == "timeout":
            raise subprocess.TimeoutExpired("private command", 60)
        if mode == "failure":
            raise RuntimeError("private /path/to/tool")
        poses = (
            []
            if mode == "empty"
            else [qvina.QvinaPose(1, math.nan if mode == "nonfinite" else 0.0), qvina.QvinaPose(2, -7.5)]
        )
        return qvina.QvinaResult(poses, poses[0] if poses else None, "", "", "", [], None)

    monkeypatch.setattr(qvina, "qvina_score_from_mol", dock)
    result = evaluate("CCC", request)
    assert result.druglikeness.validity.value is True
    if expected is not None:
        assert result.docking.status == "failed" and result.docking.error.code == expected
        assert "value" not in result.docking.model_dump()
    else:
        assert result.docking.value.affinity_kcal_mol == -7.5
        assert result.docking.value.num_poses == 2
