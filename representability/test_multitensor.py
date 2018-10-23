import numpy as np
from representability.tensor import Tensor
from representability.multitensor import MultiTensor, DualBasisElement, DualBasis

def test_multitensor_init():
    """
    Testing the generation of a multitensor object with random tensors
    """
    a = np.random.random((5, 5))
    b = np.random.random((4, 4))
    c = np.random.random((3, 3))
    at = Tensor(a, name='a')
    bt = Tensor(b, name='b')
    ct = Tensor(c, name='c')
    mt = MultiTensor([at, bt, ct])
    assert len(mt.dual_basis) == 0
    vec = np.vstack((at.vectorize(), bt.vectorize()))
    vec = np.vstack((vec, ct.vectorize()))
    assert np.allclose(vec, mt.vectorize_tensors())


def test_dual_basis_element():
    """
    Test the generation and composition of dual basis elements

    :return:
    """
    # test the addition of two BasisElements should result in a basis
    de = DualBasisElement()
    de_2 = DualBasisElement()
    db_0 = de + de_2
    assert isinstance(db_0, DualBasis)
    db_1 = db_0 + db_0
    assert isinstance(db_1, DualBasis)

    dim = 2
    opdm = np.random.random((dim, dim))
    opdm = (opdm.T + opdm)/2
    opdm = Tensor(opdm, name='opdm')
    rdm = MultiTensor([opdm])

    def generate_dual_basis_element(i, j):
        element = DualBasisElement(["opdm"], [(i, j)],
                                   [-1.0], 1 if i == j else 0, 0)
        return element

    opdm_to_oqdm_map = DualBasis()
    for val, idx in opdm.all_iterator():
        i, j = idx
        opdm_to_oqdm_map += generate_dual_basis_element(i, j)

    rdm.dual_basis = opdm_to_oqdm_map
    A, b, c = rdm.synthesize_dual_basis()
    Adense = A.todense()
    opdm_flat = opdm.data.reshape((-1, 1))
    oqdm = Adense.dot(opdm_flat)
    test_oqdm = oqdm + b.todense()
    assert np.allclose(test_oqdm.reshape((dim, dim)), np.eye(dim) - opdm.data)

def test_simplify():
    i, j, k, l = 0, 1, 2, 3
    names = ['opdm'] * 3 + ['oqdm']
    elements = [(i, j), (i, j), (i, l), (l, k)]
    coeffs = [1, 1, 0.25, 0.3]
    dbe = DualBasisElement(tensor_names=names, tensor_elements=elements, tensor_coeffs=coeffs)
    dbe.simplify()
    assert len(dbe.primal_tensors_names) == 3
    assert set(dbe.primal_coeffs) == {2, 0.25, 0.3}
    assert set(dbe.primal_tensors_names) == {'opdm', 'oqdm'}
    assert set(dbe.primal_elements) == {(0, 1), (0, 3), (3, 2)}

def test_add_element():
    """
    adding dual information to an existing element
    """
    dbe = DualBasisElement()
    dbe.add_element('cckk', (1, 2, 3, 4), 0.5)
    dbe.constant_bias = 0.33

    assert dbe.primal_coeffs == [0.5]
    assert dbe.primal_tensors_names == ['cckk']
    assert dbe.primal_elements == [(1, 2, 3, 4)]

    dbe.add_element('ck', (0, 1), 1)
    assert dbe.primal_coeffs == [0.5, 1]
    assert dbe.primal_tensors_names == ['cckk', 'ck']
    assert dbe.primal_elements == [(1, 2, 3, 4), (0, 1)]

    dbe2 = DualBasisElement()
    dbe2.constant_bias = 0.25
    dbe2.add_element('cckk', (1, 2, 3, 4), 0.5)
    dbe3 = dbe2.join_elements(dbe)
    assert np.isclose(dbe3.constant_bias, 0.58)

    assert set(dbe3.primal_elements) == {(0, 1), (1, 2, 3, 4)}
    assert np.allclose(dbe3.primal_coeffs, [1, 1])
    assert set(dbe3.primal_tensors_names) == {'ck', 'cckk'}
