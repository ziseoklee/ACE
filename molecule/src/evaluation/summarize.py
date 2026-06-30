from collections.abc import Iterable

import numpy as np
import pandas as pd

DRUGLIKENESS_COLUMNS = ["validity", "QED", "SA", "LogP", "Lipinski"]
DOCKING_COLUMNS = ["docking_success", "qvina_affinity"]


def summarize_samples(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_df = summarize_by_task(sample_df)
    summary_df = summarize_run(sample_df, task_df)
    return task_df, summary_df


def summarize_by_task(sample_df: pd.DataFrame) -> pd.DataFrame:
    if "task_id" not in sample_df:
        return pd.DataFrame()

    rows = []
    for task_id, group in sample_df.groupby("task_id", sort=True):
        row = {
            "task_id": task_id,
            "num_samples": len(group),
            "num_existing_samples": int(group["exists"].sum()) if "exists" in group else len(group),
            "ref_length": _first_or_nan(group, "ref_length"),
            "protein_filename": _first_or_nan(group, "protein_filename"),
            "reference_affinity": _first_or_nan(group, "reference_affinity"),
        }
        if _has_columns(group, DRUGLIKENESS_COLUMNS):
            row.update(_summarize_task_druglikeness(group))
        if _has_columns(group, DOCKING_COLUMNS):
            row.update(_summarize_task_docking(group))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_run(sample_df: pd.DataFrame, task_df: pd.DataFrame | None = None) -> pd.DataFrame:
    summary: dict[str, float] = {
        "num_tasks": float(sample_df["task_id"].nunique()) if "task_id" in sample_df else 0.0,
        "num_samples": float(len(sample_df)),
    }
    if "exists" in sample_df:
        summary["num_existing_samples"] = float(sample_df["exists"].sum())
        summary["missing_sample_rate"] = _safe_mean((~sample_df["exists"].astype(bool)).astype(float))

    if _has_columns(sample_df, DRUGLIKENESS_COLUMNS):
        summary.update(_summarize_run_druglikeness(sample_df))

    if _has_columns(sample_df, DOCKING_COLUMNS):
        if task_df is None:
            task_df = summarize_by_task(sample_df)
        summary.update(_summarize_run_docking(sample_df, task_df))

    return pd.DataFrame([summary])


def _summarize_task_druglikeness(group: pd.DataFrame) -> dict[str, float]:
    valid = group["validity"].astype(float)
    valid_group = group[valid == 1.0]
    return {
        "validity_mean": _safe_mean(valid),
        "QED_mean_valid": _safe_mean(valid_group["QED"]),
        "SA_mean_valid": _safe_mean(valid_group["SA"]),
        "LogP_mean_valid": _safe_mean(valid_group["LogP"]),
        "Lipinski_mean_valid": _safe_mean(valid_group["Lipinski"]),
    }


def _summarize_task_docking(group: pd.DataFrame) -> dict[str, float]:
    scores = np.sort(group["qvina_affinity"].astype(float).to_numpy())
    reference_affinity = _first_or_nan(group, "reference_affinity")
    optimized_rate = np.nan
    if not np.isnan(reference_affinity) and len(scores) > 0:
        optimized_rate = float((scores <= reference_affinity).sum() / len(scores))

    return {
        "qvina_best": _safe_index(scores, 0),
        "qvina_top3_mean": _safe_mean(scores[:3]),
        "qvina_worst": _safe_index(scores, -1),
        "qvina_mean": _safe_mean(scores),
        "qvina_median": _safe_median(scores),
        "qvina_std": _safe_std(scores),
        "qvina_optimized_rate": optimized_rate,
        "qvina_complete_fail": float(len(scores) == 0 or scores[0] > -0.01),
    }


def _summarize_run_druglikeness(sample_df: pd.DataFrame) -> dict[str, float]:
    validity = sample_df["validity"].astype(float)
    valid_df = sample_df[validity == 1.0]
    summary = {
        "validity_mean": _safe_mean(validity),
        "validity_std": _safe_std(validity),
    }
    for column in ["QED", "SA", "Lipinski"]:
        values = valid_df[column].astype(float).to_numpy()
        summary[f"{column}_mean"] = _safe_mean(values)
        summary[f"{column}_std"] = _safe_std(values)
        summary.update(_top_fraction_thresholds(values, prefix=column))
    if "LogP" in valid_df:
        summary["LogP_mean"] = _safe_mean(valid_df["LogP"])
        summary["LogP_std"] = _safe_std(valid_df["LogP"])
    return summary


def _summarize_run_docking(sample_df: pd.DataFrame, task_df: pd.DataFrame) -> dict[str, float]:
    scores = sample_df["qvina_affinity"].astype(float).to_numpy()
    sorted_scores = np.sort(scores)
    return {
        "qvina_complete_fail_rate": _safe_mean(task_df["qvina_complete_fail"]),
        "qvina_fail_rate": _safe_mean((sample_df["qvina_affinity"].astype(float) > -0.01).astype(float)),
        "qvina_best_mean": _safe_mean(task_df["qvina_best"]),
        "qvina_best_std": _safe_std(task_df["qvina_best"]),
        "qvina_top3_mean": _safe_mean(task_df["qvina_top3_mean"]),
        "qvina_top3_std": _safe_std(task_df["qvina_top3_mean"]),
        "qvina_optimized_rate_mean": _safe_mean(task_df["qvina_optimized_rate"]),
        "qvina_optimized_rate_std": _safe_std(task_df["qvina_optimized_rate"]),
        "qvina_worst_mean": _safe_mean(task_df["qvina_worst"]),
        "qvina_worst_std": _safe_std(task_df["qvina_worst"]),
        "qvina_mean": _safe_mean(task_df["qvina_mean"]),
        "qvina_mean_std": _safe_std(task_df["qvina_mean"]),
        "qvina_median_mean": _safe_mean(task_df["qvina_median"]),
        "qvina_median_std": _safe_std(task_df["qvina_median"]),
        "qvina_std_mean": _safe_mean(task_df["qvina_std"]),
        "qvina_std_std": _safe_std(task_df["qvina_std"]),
        "qvina_top25p_all_scores": _safe_quantile(sorted_scores, 0.25),
        "qvina_top50p_all_scores": _safe_quantile(sorted_scores, 0.50),
        "qvina_top75p_all_scores": _safe_quantile(sorted_scores, 0.75),
        "qvina_top100p_all_scores": _safe_index(sorted_scores, -1),
    }


def _top_fraction_thresholds(values: Iterable[float], prefix: str) -> dict[str, float]:
    sorted_values = np.sort(np.asarray(list(values), dtype=float))
    return {
        f"{prefix}_top25p": _top_fraction_threshold(sorted_values, 0.25),
        f"{prefix}_top50p": _top_fraction_threshold(sorted_values, 0.50),
        f"{prefix}_top75p": _top_fraction_threshold(sorted_values, 0.75),
        f"{prefix}_top100p": _top_fraction_threshold(sorted_values, 1.00),
    }


def _top_fraction_threshold(sorted_values: np.ndarray, fraction: float) -> float:
    if len(sorted_values) == 0:
        return np.nan
    index_from_end = max(1, int(fraction * len(sorted_values)))
    return float(sorted_values[-index_from_end])


def _safe_quantile(values: Iterable[float], quantile: float) -> float:
    values = _finite_array(values)
    if len(values) == 0:
        return np.nan
    return float(np.quantile(values, quantile))


def _safe_index(values: Iterable[float], index: int) -> float:
    values = _finite_array(values)
    if len(values) == 0:
        return np.nan
    return float(values[index])


def _safe_mean(values: Iterable[float]) -> float:
    values = _finite_array(values)
    return float(np.mean(values)) if len(values) > 0 else np.nan


def _safe_median(values: Iterable[float]) -> float:
    values = _finite_array(values)
    return float(np.median(values)) if len(values) > 0 else np.nan


def _safe_std(values: Iterable[float]) -> float:
    values = _finite_array(values)
    return float(np.std(values)) if len(values) > 0 else np.nan


def _finite_array(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    return values[np.isfinite(values)]


def _first_or_nan(group: pd.DataFrame, column: str):
    if column not in group or len(group[column]) == 0:
        return np.nan
    return group[column].iloc[0]


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in df.columns for column in columns)
