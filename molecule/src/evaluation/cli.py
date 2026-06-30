import json
from pathlib import Path
from typing import Any

import click

from evaluation.crossdocked import evaluate_crossdocked_run
from evaluation.metrics.docking import evaluate_docking_sdf
from evaluation.metrics.druglikeness import evaluate_ligand_sdf


@click.group()
def main() -> None:
    """Evaluate generated molecular samples."""


@main.command()
@click.option("--ligand_sdf", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--output_json", type=click.Path(path_type=Path, dir_okay=False), default=None)
def druglikeness(ligand_sdf: Path, output_json: Path | None) -> None:
    result: dict[str, Any] = {"ligand_sdf": str(ligand_sdf)}
    result.update(evaluate_ligand_sdf(ligand_sdf))
    _write_json_result(result, output_json)


@main.command()
@click.option("--pocket_pdb", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--ligand_sdf", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--ref_ligand_sdf", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--exhaustiveness", type=int, default=8, show_default=True)
@click.option("--num_modes", type=int, default=9, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--pad", type=float, default=8.0, show_default=True)
@click.option("--output_json", type=click.Path(path_type=Path, dir_okay=False), default=None)
def docking(
    pocket_pdb: Path,
    ligand_sdf: Path,
    ref_ligand_sdf: Path | None,
    exhaustiveness: int,
    num_modes: int,
    seed: int,
    pad: float,
    output_json: Path | None,
) -> None:
    result: dict[str, Any] = {
        "pocket_pdb": str(pocket_pdb),
        "ligand_sdf": str(ligand_sdf),
        "ref_ligand_sdf": str(ref_ligand_sdf) if ref_ligand_sdf is not None else None,
    }
    result.update(
        evaluate_docking_sdf(
            pocket_pdb,
            ligand_sdf,
            ref_ligand_sdf=ref_ligand_sdf,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            seed=seed,
            pad=pad,
        )
    )
    _write_json_result(result, output_json)


@main.command()
@click.option("--run_dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--data_root", type=click.Path(path_type=Path, file_okay=False), default=Path("data/crossdocked"))
@click.option("--metrics", multiple=True, default=("druglikeness",), show_default=True)
@click.option("--expected_num_samples", type=int, default=None)
@click.option("--output_dir", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option("--max_ligand_atoms", type=int, default=29, show_default=True)
@click.option("--docking_exhaustiveness", type=int, default=8, show_default=True)
@click.option("--docking_num_modes", type=int, default=9, show_default=True)
@click.option("--docking_seed", type=int, default=42, show_default=True)
@click.option("--docking_pad", type=float, default=8.0, show_default=True)
def crossdocked(
    run_dir: Path,
    data_root: Path,
    metrics: tuple[str, ...],
    expected_num_samples: int | None,
    output_dir: Path | None,
    max_ligand_atoms: int,
    docking_exhaustiveness: int,
    docking_num_modes: int,
    docking_seed: int,
    docking_pad: float,
) -> None:
    output_dir = output_dir or run_dir / "evaluation"
    result = evaluate_crossdocked_run(
        run_dir,
        data_root,
        metrics=_parse_metrics(metrics),
        expected_num_samples=expected_num_samples,
        output_dir=output_dir,
        max_ligand_atoms=max_ligand_atoms,
        docking_exhaustiveness=docking_exhaustiveness,
        docking_num_modes=docking_num_modes,
        docking_seed=docking_seed,
        docking_pad=docking_pad,
    )
    click.echo(f"samples_csv: {output_dir / 'samples.csv'}")
    click.echo(f"tasks_csv: {output_dir / 'tasks.csv'}")
    click.echo(f"summary_csv: {output_dir / 'summary.csv'}")
    click.echo(result.summary_df.to_string(index=False))


def _parse_metrics(metrics: tuple[str, ...]) -> list[str]:
    parsed = []
    for metric in metrics:
        parsed.extend(part for part in metric.split(",") if part)
    return parsed


def _write_json_result(result: dict[str, Any], output_json: Path | None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(payload + "\n")
    click.echo(payload)


if __name__ == "__main__":
    main()
