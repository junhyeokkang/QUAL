"""User-facing entry points: `combine` and `align`."""
from __future__ import annotations

import numpy as np

from .core import (DEFAULT_LEVELS, HUBER_ITERATIONS, HUBER_THRESHOLD,
                   align_quantiles, qual_combine_stacked)


def _stack_members(forecasts) -> tuple[np.ndarray, bool]:
    """Return a (K, n, Q, T) float array and whether the input was a single instance.

    Accepts a dict or list of per-model arrays, each (Q, T) or (n, Q, T),
    or an already stacked (K, Q, T) or (K, n, Q, T) array.
    """
    if isinstance(forecasts, dict):
        members = [np.asarray(value, dtype=np.float64) for value in forecasts.values()]
    elif isinstance(forecasts, (list, tuple)):
        members = [np.asarray(value, dtype=np.float64) for value in forecasts]
    else:
        stacked = np.asarray(forecasts, dtype=np.float64)
        if stacked.ndim == 3:
            return stacked[:, None], True
        if stacked.ndim == 4:
            return stacked, False
        raise ValueError(f"stacked forecasts must be (K, Q, T) or (K, n, Q, T); got {stacked.shape}")

    if len(members) < 2:
        raise ValueError(f"need at least two member forecasts; got {len(members)}")
    shapes = {member.shape for member in members}
    if len(shapes) > 1:
        raise ValueError(f"member forecasts must share one shape; got {sorted(shapes)}")
    stacked = np.stack(members, axis=0)
    if stacked.ndim == 3:
        return stacked[:, None], True
    if stacked.ndim == 4:
        return stacked, False
    raise ValueError(f"each member must be (Q, T) or (n, Q, T); got {members[0].shape}")


def _check_levels(levels, num_levels: int) -> np.ndarray:
    if levels is None:
        if num_levels != len(DEFAULT_LEVELS):
            raise ValueError(f"forecasts have {num_levels} quantile rows; pass levels= explicitly")
        return DEFAULT_LEVELS
    levels = np.asarray(levels, dtype=np.float64)
    if levels.shape != (num_levels,):
        raise ValueError(f"levels must have length {num_levels}; got shape {levels.shape}")
    if not (np.all(np.diff(levels) > 0) and levels[0] > 0 and levels[-1] < 1):
        raise ValueError("levels must be strictly increasing and inside (0, 1)")
    return levels


def _prepare(forecasts, levels):
    stacked, single_instance = _stack_members(forecasts)
    if stacked.shape[0] < 2:
        raise ValueError("need at least two member forecasts")
    if not np.isfinite(stacked).all():
        raise ValueError("forecasts contain NaN or inf values")
    levels = _check_levels(levels, stacked.shape[2])
    # Each member must be a valid quantile function before its CDF is built.
    stacked = np.sort(stacked, axis=2)
    return stacked, levels, single_instance


def combine(forecasts, levels=None, threshold: float = HUBER_THRESHOLD,
            num_iterations: int = HUBER_ITERATIONS) -> np.ndarray:
    """Combine quantile forecasts from several models with QUAL.

    Args:
        forecasts: dict or list of per-model arrays, each (Q, T) for one forecast
            instance or (n, Q, T) for a batch; or a stacked (K, Q, T) / (K, n, Q, T)
            array. Quantile rows go from the lowest level to the highest.
        levels: probability level of each quantile row. Defaults to 0.1, ..., 0.9.
        threshold: Huber threshold c in units of the MAD scale.
        num_iterations: reweighting steps of the Huber center.

    Returns:
        Combined quantiles, (Q, T) for a single instance or (n, Q, T) for a batch.
    """
    stacked, levels, single_instance = _prepare(forecasts, levels)
    combined = qual_combine_stacked(stacked, levels, threshold, num_iterations)
    return combined[0] if single_instance else combined


def align(forecasts, levels=None) -> np.ndarray:
    """Return each member forecast after quantile alignment, without aggregation.

    Returns:
        (K, Q, T) for a single instance or (K, n, Q, T) for a batch, in the
        member order of the input.
    """
    stacked, levels, single_instance = _prepare(forecasts, levels)
    aligned = align_quantiles(stacked, levels)
    return aligned[:, 0] if single_instance else aligned
