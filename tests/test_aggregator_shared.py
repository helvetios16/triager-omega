import numpy as np
import pytest

from triager_omega.modules.aggregator import (
    gate_mask, linear_fuse, make_fuser, rrf_fuse, rrf_rank_term,
)


def _independent_rrf(matrix, k):
    result = np.zeros(matrix.shape, dtype=float)
    # Deliberately reproduce NumPy's default argsort ordering (not stable sort).
    for i, row in enumerate(matrix):
        order = np.argsort(-row)
        for rank, col in enumerate(order):
            result[i, col] = 1 / (k + rank + 1)
    return result


def test_shared_linear_fusion_and_gate_are_explicit():
    nps = np.array([[.9, .8, .1], [.9, .2, 0.]])
    nis = np.array([[.2, 1., 0.], [1., .1, .4]])
    expected_mask = np.array([[1.], [0.]])
    np.testing.assert_array_equal(gate_mask(nps, .2), expected_mask)
    np.testing.assert_allclose(linear_fuse(nps, nis, .5, .2), nps + .5 * nis * expected_mask)
    np.testing.assert_allclose(make_fuser(nps, nis, "linear", 60, .2)(.5), nps + .5 * nis * expected_mask)


def test_shared_rrf_preserves_rank_ties_and_positive_nis_mask():
    nps = np.array([[.8, .8, .2]])
    nis = np.array([[0., .5, .5]])
    expected_nps = _independent_rrf(nps, 10)
    expected_nis = _independent_rrf(nis, 10) * (nis > 0)
    np.testing.assert_allclose(rrf_rank_term(nps, 10), expected_nps)
    np.testing.assert_allclose(rrf_fuse(nps, nis, .7, 10), expected_nps + .7 * expected_nis)
    np.testing.assert_allclose(make_fuser(nps, nis, "rrf", 10)(.7), expected_nps + .7 * expected_nis)


def test_shared_fusion_rejects_misaligned_matrices():
    with pytest.raises(ValueError, match="alineadas"):
        linear_fuse(np.zeros((2, 3)), np.zeros((2, 2)), 1.)
