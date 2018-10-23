from itertools import product
import numpy as np
from representability.purification.higham import heaviside, higham_polynomial, higham_root, \
                                                 fixed_trace_positive_projection, map_to_tensor, map_to_matrix


def test_heaviside():
    assert np.isclose(heaviside(0), 1.0)
    assert np.isclose(heaviside(0.5), 1.0)
    assert np.isclose(heaviside(-0.5), 0.0)
    assert np.isclose(heaviside(-0.5, -1), 1.0)
    assert np.isclose(heaviside(-2, -1), 0)


def test_highham_polynomial():
    eigs = np.arange(10)
    assert np.isclose(higham_polynomial(eigs, eigs[-1]), 0.0)
    assert np.isclose(higham_polynomial(eigs, 0), sum(eigs))
    assert np.isclose(higham_polynomial(eigs, 5), sum(eigs[5:] - 5))
    assert np.isclose(higham_polynomial(eigs, 8), sum(eigs[8:] - 8))


def test_higham_root():
    dim = 20
    np.random.seed(42)
    mat = np.random.random((dim, dim))
    mat = 0.5 * (mat + mat.T)
    w, v = np.linalg.eigh(mat)
    target_trace = np.round(w[-1]-1)
    sigma = higham_root(w, target_trace)
    assert np.isclose(higham_polynomial(w, shift=sigma), target_trace)


def test_matrix_2_tesnor():
    dim = 10
    np.random.seed(42)
    mat = np.random.random((dim**2, dim**2))
    mat = 0.5 * (mat + mat.T)
    tensor = map_to_tensor(mat)
    for p, q, r, s in product(range(dim), repeat=4):
        assert np.isclose(tensor[p, q, r, s], mat[p * dim + q, r * dim + s])

    test_mat = map_to_matrix(tensor)
    assert np.allclose(test_mat, mat)


def test_reconstruction():
    dim = 20
    np.random.seed(42)
    mat = np.random.random((dim, dim))
    mat = 0.5 * (mat + mat.T)
    test_mat = np.zeros_like(mat)
    w, v = np.linalg.eigh(mat)
    for i in range(w.shape[0]):
        test_mat += w[i] * v[:, [i]].dot(v[:, [i]].T)
    assert np.allclose(test_mat - mat, 0.0)

    test_mat = fixed_trace_positive_projection(mat, np.trace(mat))
    assert np.isclose(np.trace(test_mat), np.trace(mat))
    w, v = np.linalg.eigh(test_mat)
    assert np.all(w >= -(float(4.0E-15)))


def test_mlme():
    """
    Test from fig 1 of maximum likelihood minimum effort

    :return:
    """
    eigs = np.array(list(reversed([3.0/5, 1.0/2, 7.0/20, 1.0/10, -11.0/20])))
    target_trace = 1.0
    sigma = higham_root(eigs, target_trace)
    shifted_eigs = np.multiply(heaviside(eigs - sigma), (eigs - sigma))
    assert np.allclose(shifted_eigs, [0, 0, 1.0/5, 7.0/20, 9.0/20])
