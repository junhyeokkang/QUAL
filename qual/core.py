"""QUAL: quantile alignment, Huber aggregation, and rearrangement.

Shape convention for stacked forecasts: (K, n, Q, T) with
K members, n forecast instances, Q quantile levels (ascending), T horizon steps.
"""
from __future__ import annotations

import numpy as np

DEFAULT_LEVELS = np.arange(1, 10, dtype=np.float64) / 10
HUBER_THRESHOLD = 1.345
HUBER_ITERATIONS = 2
MAD_TO_STD = 1.4826


def _interpolate_monotone(x_knots: np.ndarray, y_knots: np.ndarray,
                          x_query: np.ndarray, min_gap: float) -> np.ndarray:
    """Piecewise-linear interpolation of a monotone curve, batched over leading axes.

    Knots run along the last axis. Queries outside the knot range are clipped
    to the nearest end knot.
    """
    num_knots = x_knots.shape[-1]
    knots_at_or_below = (x_knots[..., :, None] <= x_query[..., None, :]).sum(axis=-2)
    left = np.clip(knots_at_or_below - 1, 0, num_knots - 2)
    right = left + 1

    y_knots = np.broadcast_to(y_knots, x_knots.shape)
    x_low = np.take_along_axis(x_knots, left, axis=-1)
    x_high = np.take_along_axis(x_knots, right, axis=-1)
    y_low = np.take_along_axis(y_knots, left, axis=-1)
    y_high = np.take_along_axis(y_knots, right, axis=-1)

    fraction = np.clip((x_query - x_low) / np.maximum(x_high - x_low, min_gap), 0.0, 1.0)
    return y_low + fraction * (y_high - y_low)


def align_quantiles(forecasts: np.ndarray, levels: np.ndarray = DEFAULT_LEVELS) -> np.ndarray:
    """Re-express every member forecast on the pooled probability reference.

    Each member's quantiles define a piecewise-linear CDF. Their equally
    weighted mixture is the pooled CDF. Every reported value is relabelled by
    its pooled percentile, and each member's (percentile, value) curve is read
    back at the requested levels.

    Args:
        forecasts: (K, n, Q, T) member quantiles, sorted along the Q axis.
        levels: (Q,) probability levels of the quantile rows.

    Returns:
        (K, n, Q, T) aligned member quantiles.
    """
    num_members, num_instances, num_levels, horizon = forecasts.shape
    levels = np.asarray(levels, dtype=np.float64)

    # (K, n, Q, T) -> (n, T, K, Q): instances and horizon steps become batch axes.
    values = forecasts.transpose(1, 3, 0, 2).astype(np.float64)
    all_member_values = values.reshape(num_instances, horizon, num_members * num_levels)

    member_cdfs = [
        _interpolate_monotone(values[:, :, member, :], levels, all_member_values, min_gap=1e-12)
        for member in range(num_members)
    ]
    pooled_percentiles = (sum(member_cdfs) / num_members).reshape(
        num_instances, horizon, num_members, num_levels)

    requested_levels = np.broadcast_to(levels, pooled_percentiles.shape)
    aligned = _interpolate_monotone(pooled_percentiles, values, requested_levels, min_gap=1e-9)
    return aligned.transpose(2, 0, 3, 1)


def huber_aggregate(forecasts: np.ndarray, threshold: float = HUBER_THRESHOLD,
                    num_iterations: int = HUBER_ITERATIONS) -> np.ndarray:
    """Huber robust center over the member axis, computed separately for every cell.

    Starts from the median and the MAD scale, then runs iteratively reweighted
    averaging. A member keeps weight 1 where its standardized distance from the
    center is at most `threshold`, and weight threshold / distance beyond it.

    Args:
        forecasts: (K, n, Q, T) member quantiles.

    Returns:
        (n, Q, T) robust centers.
    """
    center = np.median(forecasts, axis=0)
    scale = MAD_TO_STD * np.median(np.abs(forecasts - center), axis=0) + 1e-12
    for _ in range(num_iterations):
        distance = np.abs(forecasts - center) / scale
        weights = np.minimum(1.0, threshold / np.maximum(distance, 1e-12))
        center = (weights * forecasts).sum(axis=0) / weights.sum(axis=0)
    return center


def rearrange_quantiles(quantiles: np.ndarray) -> np.ndarray:
    """Sort along the quantile axis so the combined forecast never crosses.

    Args:
        quantiles: (n, Q, T) combined quantiles.
    """
    return np.sort(quantiles, axis=1)


def qual_combine_stacked(forecasts: np.ndarray, levels: np.ndarray = DEFAULT_LEVELS,
                         threshold: float = HUBER_THRESHOLD,
                         num_iterations: int = HUBER_ITERATIONS) -> np.ndarray:
    """Full QUAL pipeline on a stacked (K, n, Q, T) array; returns (n, Q, T)."""
    aligned = align_quantiles(forecasts, levels)
    centers = huber_aggregate(aligned, threshold, num_iterations)
    return rearrange_quantiles(centers)
