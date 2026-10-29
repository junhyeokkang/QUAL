import numpy as np
import pytest

from qual import align, combine

LEVELS = np.arange(1, 10) / 10


def random_quantiles(rng, shape):
    return np.sort(rng.normal(size=shape), axis=-2)


def test_identical_members_are_unchanged():
    rng = np.random.default_rng(0)
    member = random_quantiles(rng, (9, 12))
    forecasts = [member, member.copy(), member.copy()]
    np.testing.assert_allclose(align(forecasts), np.stack(forecasts), atol=1e-9)
    np.testing.assert_allclose(combine(forecasts), member, atol=1e-9)


def test_output_shapes_follow_input():
    rng = np.random.default_rng(1)
    single = {name: random_quantiles(rng, (9, 7)) for name in "abc"}
    batch = [random_quantiles(rng, (5, 9, 7)) for _ in range(4)]
    assert combine(single).shape == (9, 7)
    assert combine(batch).shape == (5, 9, 7)
    assert align(batch).shape == (4, 5, 9, 7)


def test_combined_quantiles_do_not_cross():
    rng = np.random.default_rng(2)
    forecasts = [random_quantiles(rng, (20, 9, 10)) * scale for scale in (0.5, 1.0, 3.0)]
    combined = combine(forecasts)
    assert (np.diff(combined, axis=1) >= 0).all()


def test_custom_levels_require_matching_length():
    rng = np.random.default_rng(3)
    forecasts = [random_quantiles(rng, (5, 4)) for _ in range(3)]
    with pytest.raises(ValueError):
        combine(forecasts)
    assert combine(forecasts, levels=[0.1, 0.25, 0.5, 0.75, 0.9]).shape == (5, 4)


def test_rejects_single_member_and_nan():
    rng = np.random.default_rng(4)
    member = random_quantiles(rng, (9, 3))
    with pytest.raises(ValueError):
        combine([member])
    broken = member.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError):
        combine([member, broken])
