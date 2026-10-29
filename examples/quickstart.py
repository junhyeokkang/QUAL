"""Combine three synthetic quantile forecasts with QUAL.

Replace the synthetic arrays with the quantile outputs of your own models,
each shaped (Q, T) or (n, Q, T) on the levels 0.1, ..., 0.9.
"""
import numpy as np

from qual import align, combine

LEVELS = np.arange(1, 10) / 10
HORIZON = 24


def synthetic_forecast(rng, location, spread):
    """Gaussian quantiles around a sine path, standing in for one model's output."""
    path = location + np.sin(np.linspace(0, 2 * np.pi, HORIZON))
    standard_quantiles = np.array([-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28])
    noise = rng.normal(0, 0.05, size=HORIZON)
    return path[None, :] + noise[None, :] + spread * standard_quantiles[:, None]


rng = np.random.default_rng(0)
forecasts = {
    "model_a": synthetic_forecast(rng, location=0.0, spread=0.5),
    "model_b": synthetic_forecast(rng, location=0.2, spread=1.0),
    "model_c": synthetic_forecast(rng, location=-0.1, spread=0.3),
}

combined = combine(forecasts, levels=LEVELS)
aligned = align(forecasts, levels=LEVELS)

print("combined forecast:", combined.shape)    # (9, 24)
print("aligned members:  ", aligned.shape)     # (3, 9, 24)
print("median of the combined forecast, first 5 steps:", combined[4, :5].round(3))
