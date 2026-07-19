"""Small, dependency-free helpers for ACE exponent and resampling schedules."""

from __future__ import annotations


def validate_unit_interval(value: float, *, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}.")
    return value


def quadratic_bump(progress: float, strength: float) -> float:
    """Return ``B t (1-t)`` for generation progress ``t`` in ``[0, 1]``."""
    progress = validate_unit_interval(progress, name="progress")
    return float(strength) * progress * (1.0 - progress)


def quadratic_bump_derivative(progress: float, strength: float) -> float:
    """Return the continuous-time derivative ``B (1-2t)``.

    This derivative must *not* be divided by the number of solver steps.  The
    Euler update supplies the single ``dt`` factor.
    """
    progress = validate_unit_interval(progress, name="progress")
    return float(strength) * (1.0 - 2.0 * progress)


def progress_to_step(progress: float, steps: int) -> int:
    """Map continuous generation progress to the closest zero-based step."""
    progress = validate_unit_interval(progress, name="progress")
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}.")
    return min(steps - 1, round(progress * steps))


def scheduled_steps(progress_values: list[float] | tuple[float, ...], steps: int) -> tuple[int, ...]:
    """Return sorted, unique solver steps for scheduled resampling."""
    return tuple(sorted({progress_to_step(value, steps) for value in progress_values}))
