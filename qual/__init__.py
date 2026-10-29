"""QUAL: label-free quantile alignment for combining time-series foundation models."""
from .api import align, combine
from .core import align_quantiles, huber_aggregate, qual_combine_stacked, rearrange_quantiles

__all__ = ["combine", "align", "align_quantiles", "huber_aggregate",
           "rearrange_quantiles", "qual_combine_stacked"]
__version__ = "0.1.0"
