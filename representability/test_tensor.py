from itertools import product
import numpy as np
from representability.tensor import Tensor, Bijection, index_index_basis, \
    index_tuple_basis


def test_bijection():
    """
    Testing the basis bijection
    """
    b = Bijection(lambda x: x + 1, lambda y: y - 1, lambda: (1, 1))
    assert b.fwd(2) == 3
    assert b.rev(2) == 1
    assert b.fwd(b.rev(5)) == 5
    assert b.domain_element_sizes() == (1, 1)


def test_index_basis():
    b = index_index_basis(5)
    assert b.fwd(4) == 4
    assert b.rev(4) == 4
    assert b.domain_element_sizes() == (1, 1)


def test_geminal_basis():
    gems = list(product(range(5), repeat=2))
    b = index_tuple_basis(gems)
    assert b.fwd(4) == (0, 4)
    assert b.rev((0, 4)) == 4
    assert b.rev(b.fwd(4)) == 4
    assert b.domain_element_sizes() == (1, 2)


def test_tensor_index():
    """
    Testing the index_vectorized which is mapped through the basis

    :return:
    """
    a = np.arange(16).reshape((4, 4), order='C')
    basis = [(0, 0), (0, 1), (1, 0), (1, 1)]
    basis = index_tuple_basis(basis)
    tt = Tensor(a, basis=basis)
    assert np.allclose(tt.data, a)
    assert tt.size == 16
    assert isinstance(tt.basis, Bijection)
    assert np.isclose(tt.index_vectorized(0, 0, 0, 0), 0)
    assert np.isclose(tt.index_vectorized(0, 0, 0, 1), 1)
    assert np.isclose(tt.index_vectorized(0, 0, 1, 0), 2)
    assert np.isclose(tt.index_vectorized(0, 0, 1, 1), 3)
    assert np.isclose(tt.index_vectorized(0, 1, 0, 0), 4)
    assert np.isclose(tt.index_vectorized(0, 1, 0, 1), 5)
    assert np.isclose(tt.index_vectorized(0, 1, 1, 0), 6)
    assert np.isclose(tt.index_vectorized(0, 1, 1, 1), 7)
    assert np.isclose(tt.index_vectorized(1, 0, 0, 0), 8)
    # etc...

    a = np.arange(16).reshape((4, 4), order='C')
    tt = Tensor(a)  # the canonical basis
    assert np.isclose(tt.index_vectorized(0, 0), 0)
    assert np.isclose(tt.index_vectorized(0, 1), 1)
    assert np.isclose(tt.index_vectorized(0, 2), 2)
    assert np.isclose(tt.index_vectorized(0, 3), 3)
    assert np.isclose(tt.index_vectorized(1, 0), 4)
    assert np.isclose(tt.index_vectorized(1, 1), 5)
    assert np.isclose(tt.index_vectorized(1, 2), 6)
    assert np.isclose(tt.index_vectorized(1, 3), 7)


def test_tensor_init():
    """
    Initialize a tensor and confirm that we have all values None

    Initialize a tensor and confirm that all values are equal to the correct values
    and iteration over the tensors are the correct values
    :return:
    """
    test_tensor = Tensor()
    assert test_tensor.dim is None
    assert test_tensor.ndim is None
    assert test_tensor.data is None
    assert test_tensor.size is None
    assert test_tensor.basis is None

    a = np.arange(16).reshape((4, 4))
    test_tensor = Tensor(a)
    assert np.allclose(test_tensor.data, a)
    assert test_tensor.size == 16
    assert isinstance(test_tensor.basis, Bijection)

    a_triu = a[np.triu_indices_from(a)]
    a_tril = a[np.tril_indices_from(a)]

    counter = 0
    for val, idx in test_tensor.utri_iterator():
        assert val == a[tuple(idx)]
        assert val == a_triu[counter]
        counter += 1
    assert counter == 4 * (4 + 1) / 2

    counter = 0
    for val, idx in test_tensor.ltri_iterator():
        assert val == a[tuple(idx)]
        assert val == a_tril[counter]
        counter += 1
    assert counter == 4 * (4 + 1) / 2

    assert np.allclose(test_tensor.vectorize(), a.reshape((-1, 1), order='C'))


def test_tensor_basis():
    """
    make a matrix that has a different basis than indexing
    """
    n = 4
    dim = int(n * (n - 1) / 2)
    geminals = []
    bas = {}
    cnt = 0
    for i in range(4):
        for j in range(i + 1, 4):
            bas[cnt] = (i, j)
            geminals.append((i, j))
            cnt += 1
    rev_bas = dict(zip(bas.values(), bas.keys()))

    rand_mat = np.random.random((dim, dim))
    basis_bijection = index_tuple_basis(geminals)
    test_tensor = Tensor(rand_mat, basis=basis_bijection)
    assert test_tensor.basis.fwd(0) == (0, 1)
    assert test_tensor.basis.fwd(2) == (0, 3)
    assert test_tensor.basis.rev(test_tensor.basis.fwd(5)) == 5
    assert test_tensor.ndim == 2
    assert test_tensor.dim == dim
    # index into data directly
    assert test_tensor[2, 3] == rand_mat[2, 3]
    # index into data via basis indexing
    assert test_tensor(0, 1, 0, 1) == rand_mat[0, 0]
    assert test_tensor(1, 2, 0, 1) == rand_mat[rev_bas[(1, 2)], rev_bas[(0, 1)]]
    assert test_tensor.index_vectorized(1, 2, 0, 1) == rev_bas[(1, 2)] * dim + \
                                                       rev_bas[(0, 1)]

    # testing iteration over the upper triangle
    for iter_vals in test_tensor.utri_iterator():
        val, [i, j] = iter_vals
        assert val == rand_mat[
            test_tensor.basis.rev(i), test_tensor.basis.rev(j)]

