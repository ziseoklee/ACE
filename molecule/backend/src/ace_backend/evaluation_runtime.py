"""Use existing RDKit metrics and QuickVina while preserving individual failures."""

import logging
import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from ace_backend.evaluation_schema import (
    DockingConfig,
    DockingReport,
    DockingValue,
    DruglikenessReports,
    EvaluationConfig,
    FailedReport,
    MetricGroups,
    ScaffoldValue,
    SkippedReport,
    SuccessfulReport,
)
from ace_backend.jobs_schema import Error

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _measure(function: Callable[[], T]) -> SuccessfulReport[T] | FailedReport:  # noqa: UP047 -- Python 3.11.
    try:
        value = function()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("A metric returned a non-finite value.")
        return SuccessfulReport(value=value)
    except ImportError:
        logger.exception("Metric dependency unavailable")
        return FailedReport(
            error=Error(code="metric_unavailable", message="A required metric dependency is unavailable.")
        )
    except Exception:
        logger.exception("Metric calculation failed")
        return FailedReport(error=Error(code="metric_failed", message="The metric could not be calculated."))


def _druglikeness(molecule: "Mol") -> DruglikenessReports:
    try:
        from evaluation.metrics import druglikeness
    except ImportError:
        failed = FailedReport(
            error=Error(code="metric_unavailable", message="Druglikeness dependencies are unavailable.")
        )
        return DruglikenessReports(
            validity=SuccessfulReport(value=True), qed=failed, sa_normalized=failed, logp=failed, lipinski_legacy=failed
        )

    logp = _measure(lambda: float(druglikeness.rdMolDescriptors.CalcCrippenDescriptors(molecule)[0]))
    legacy = (
        _measure(lambda: druglikeness.calculate_lipinski_legacy(molecule, logp.value))
        if isinstance(logp, SuccessfulReport)
        else FailedReport(
            error=Error(code="metric_failed", message="Lipinski scoring requires a successfully calculated LogP.")
        )
    )
    return DruglikenessReports(
        validity=SuccessfulReport(value=True),
        qed=_measure(lambda: _unit_interval(druglikeness.QED.qed(molecule))),
        sa_normalized=_measure(lambda: druglikeness.calculate_sa_normalized(molecule)),
        logp=logp,
        lipinski_legacy=legacy,
    )


def _unit_interval(value: float) -> float:
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError("Expected a metric in [0, 1].")
    return value


def _scaffold(molecule: "Mol", fragment: "Mol") -> ScaffoldValue:
    from rdkit import Chem

    scaffold = Chem.Mol(fragment)
    Chem.SanitizeMol(scaffold)
    return ScaffoldValue(
        contains_fragment=Chem.RemoveHs(molecule).HasSubstructMatch(Chem.RemoveHs(scaffold), useChirality=False)
    )


def _docking(molecule: "Mol", pocket: Path, reference: "Mol", config: DockingConfig) -> DockingReport:
    try:
        from rdkit import Chem

        from evaluation.backends.qvina import qvina_score_from_mol

        result = qvina_score_from_mol(
            str(pocket),
            Chem.Mol(molecule),
            ref_mol=Chem.Mol(reference),
            seed=config.seed,
            exhaustiveness=config.exhaustiveness,
            num_modes=config.num_modes,
            pad=config.padding_angstrom,
        )
        affinities = [pose.affinity for pose in result.poses]
        if not affinities or not all(math.isfinite(value) for value in affinities):
            raise ValueError("QuickVina returned no finite poses.")
        return SuccessfulReport(value=DockingValue(affinity_kcal_mol=min(affinities), num_poses=len(affinities)))
    except subprocess.TimeoutExpired:
        logger.exception("Docking timed out")
        return FailedReport(
            error=Error(code="docking_timeout", message="An external docking command exceeded its time limit.")
        )
    except Exception:
        logger.exception("Docking failed")
        return FailedReport(
            error=Error(code="docking_failed", message="QuickVina could not produce a valid docking result.")
        )


def evaluate_molecule(
    molecule: "Mol", config: EvaluationConfig, *, fragment: "Mol | None", pocket: Path | None, reference: "Mol | None"
) -> MetricGroups:
    from rdkit import Chem

    sanitized = Chem.Mol(molecule)
    try:
        Chem.SanitizeMol(sanitized)
    except (ValueError, RuntimeError):
        skipped = SkippedReport()
        return MetricGroups(
            druglikeness=DruglikenessReports(
                validity=SuccessfulReport(value=False),
                qed=skipped,
                sa_normalized=skipped,
                logp=skipped,
                lipinski_legacy=skipped,
            )
            if "druglikeness" in config.metrics
            else None,
            scaffold_preservation=skipped if "scaffold_preservation" in config.metrics else None,
            docking=skipped if "docking" in config.metrics else None,
        )
    scaffold = None
    docking = None
    if "scaffold_preservation" in config.metrics:
        assert fragment is not None, "Validated scaffold input is required."
        scaffold = _measure(lambda: _scaffold(sanitized, fragment))
    if "docking" in config.metrics:
        assert pocket is not None and reference is not None and config.docking is not None, (
            "Validated docking inputs are required."
        )
        docking = _docking(sanitized, pocket, reference, config.docking)
    return MetricGroups(
        druglikeness=_druglikeness(sanitized) if "druglikeness" in config.metrics else None,
        scaffold_preservation=scaffold,
        docking=docking,
    )


def has_evaluation_values(groups: MetricGroups) -> bool:
    reports = [groups.scaffold_preservation, groups.docking]
    if groups.druglikeness is not None:
        reports.extend(getattr(groups.druglikeness, name) for name in type(groups.druglikeness).model_fields)
    return any(isinstance(report, SuccessfulReport) for report in reports)
