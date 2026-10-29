# QUAL: Quantile Alignment Improves Zero-Shot Time-Series Foundation Models

Official implementation of **QUAL** (**Qu**antile **al**ignment), a label-free
framework for improving and combining quantile forecasts from several
time-series foundation models (TSFMs).

QUAL uses the model pool as a probability reference rather than as the final
forecast. It

1. builds each member's piecewise-linear CDF from its reported quantiles and
   averages them into a pooled CDF,
2. re-expresses every member forecast on this shared reference
   (**quantile alignment**),
3. combines the aligned members with a per-cell **Huber robust center**, and
4. sorts the result so the quantiles never cross (**rearrangement**).

It needs no training, no target labels, and no change to the member models.
Any model that outputs quantiles can be plugged in.

## Installation

```bash
pip install git+https://github.com/junhyeokkang/QUAL.git
```

The only dependency is NumPy.

## Usage

```python
import numpy as np
from qual import combine, align

# Quantile forecasts from your own models, each shaped (Q, T)
# for one forecast instance or (n, Q, T) for a batch.
forecasts = {
    "chronos-2": chronos_quantiles,
    "timesfm-2.5": timesfm_quantiles,
    "tirex": tirex_quantiles,
}

levels = np.arange(1, 10) / 10           # quantile level of each row
combined = combine(forecasts, levels=levels)   # QUAL forecast, (Q, T) or (n, Q, T)
aligned = align(forecasts, levels=levels)      # aligned members, (K, Q, T) or (K, n, Q, T)
```

`levels` defaults to `0.1, ..., 0.9` when each forecast has nine quantile rows.
The Huber threshold (`threshold=1.345`) and the number of reweighting steps
(`num_iterations=2`) are the values used in the paper.

A runnable example with synthetic forecasts is in
[`examples/quickstart.py`](examples/quickstart.py).

## Citation

```bibtex
@inproceedings{kang2026qual,
  title     = {{QUAL}: Quantile Alignment Improves Zero-Shot Time-Series Foundation Models},
  author    = {Kang, Junhyeok and Han, Sangjun and Seo, Jun and Lee, Seunghan and Lee, Jaehoon and Kang, Dongwan and Kim, Minjae and Yoo, Sungdong and Lim, Tae Yoon and Choi, Hwanil and Lee, Soonyoung and Ahn, Wonbin},
  booktitle = {NeurIPS 2026 Workshop on Foundation Models for Temporal Systems: From Forecasting to World Modeling},
  year      = {2026}
}
```
