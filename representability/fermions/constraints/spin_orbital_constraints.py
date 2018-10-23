"""
This module contains methods that generate the dual basis that is common in 2-RDM theory
"""
import numpy as np
from itertools import product
from representability.dualbasis import DualBasis, DualBasisElement
from representability.fermions.basis_utils import (_coord_generator,
                                                   triples_spin_orbital_antisymm_basis,
                                                   _three_parity)


def kdelta(i, j):
    return 1.0 if i == j else 0.0


def trace_constraint(dim, normalization):
    """
    Generate the trace constraint on the 2-RDM

    :param dim: spinless Fermion basis rank
    :return: the dual basis element
    :rtype: DualBasisElement
    """
    tensor_elements = [(i, j, i, j) for i, j in product(range(dim), repeat=2)]
    tensor_names = ['cckk'] * (dim**2)
    tensor_coeffs = [1.0] * (dim**2)
    bias = 0
    return DualBasisElement(tensor_names=tensor_names, tensor_elements=tensor_elements,
                            tensor_coeffs=tensor_coeffs, bias=bias, scalar=normalization)


def antisymmetry_constraints(dim):
    """
    The dual basis elements representing the antisymmetry constraints

    :param dim: spinless Fermion basis rank
    :return: the dual basis of antisymmetry_constraints
    :rtype: DualBasis
    """
    # dual_basis = DualBasis()
    dbe_list = []
    for p, q, r, s in product(range(dim), repeat=4):
        if p * dim + q <= r * dim + s:
            if p < q and r < s:
                tensor_elements = [tuple(indices) for indices in _coord_generator(p, q, r, s)]
                tensor_names = ['cckk'] * len(tensor_elements)
                tensor_coeffs = [0.5] * len(tensor_elements)
                dbe = DualBasisElement()
                for n, e, c in zip(tensor_names, tensor_elements, tensor_coeffs):
                    dbe.add_element(n, e, c)

                # dual_basis += dbe
                dbe_list.append(dbe)

    return DualBasis(elements=dbe_list)


def d2_d1_mapping(dim, normalization):
    """
    Construct dual basis for contracting d2 -> d1

    :param dim: linear dimension of the 1-RDM
    :param normalization: normalization constant for coeff of D1 elememnts
    :return: the dual basis of the contraction
    :rtype: DualBasis
    """
    db_basis = DualBasis()
    dbe_list = []
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            for r in range(dim):
                # duplicate entries get summed in DualBasisElement
                dbe.add_element('cckk', (i, r, j, r), 0.5)
                dbe.add_element('cckk', (j, r, i, r), 0.5)

            # D1 terms
            dbe.add_element('ck', (i, j), -0.5 * normalization)
            dbe.add_element('ck', (j, i), -0.5 * normalization)
            dbe.simplify()
            # db_basis += dbe
            dbe_list.append(dbe)

    return DualBasis(elements=dbe_list)  # db_basis


def d1_q1_mapping(dim):
    """
    Map the ck to kc

    D1 + Q1 = I

    :param dim: linear dimension of the 1-RDM
    :return: the dual basis of the mapping
    :rtype: DualBasis
    """
    db = DualBasis()
    dbe_list = []
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            if i != j:
                dbe.add_element('ck', (i, j), 0.5)
                dbe.add_element('ck', (j, i), 0.5)
                dbe.add_element('kc', (j, i), 0.5)
                dbe.add_element('kc', (i, j), 0.5)
                dbe.dual_scalar = 0.0
            else:
                dbe.add_element('ck', (i, j), 1.0)
                dbe.add_element('kc', (i, j), 1.0)
                dbe.dual_scalar = 1.0

            # db += dbe
            dbe_list.append(dbe)

    return DualBasis(elements=dbe_list)  # db


def d2_q2_mapping(dim):
    """
    Generate dual basis elements for d2-> q2 mapping

    :param dim: linear dimension of the 1-RDM
    :return: the dual basis of the mapping
    :rtype: DualBasis
    """
    krond = np.eye(dim)
    # db = DualBasis()
    dbe_list = []

    def d2q2element(p, q, r, s, factor):
        """
        Build the dual basis element for symmetric form of 2-marginal

        :param p: tensor index
        :param q: tensor index
        :param r: tensor index
        :param s: tensor index
        :param factor: scaling coeff for a symmetric constraint
        :return: the dual basis of the mapping
        """
        dbe = DualBasisElement()
        dbe.add_element('cckk', (p, q, r, s), -1.0 * factor)
        dbe.add_element('kkcc', (r, s, p, q), +1.0 * factor)
        dbe.add_element('ck', (p, r), krond[q, s] * factor)
        dbe.add_element('ck', (q, s), krond[p, r] * factor)
        dbe.add_element('ck', (p, s), -1. * krond[q, r] * factor)
        dbe.add_element('ck', (q, r), -1. * krond[p, s] * factor)
        dbe.dual_scalar = (krond[q, s] * krond[p, r] - krond[q, r] * krond[p, s]) * factor

        return dbe

    for p, q, r, s in product(range(dim), repeat=4):
        if p * dim + q <= r * dim + s:
            db_element = d2q2element(p, q, r, s, 0.5)
            db_element_2 = d2q2element(r, s, p, q, 0.5)
            # db += db_element.join_elements(db_element_2)
            dbe_list.append(db_element.join_elements(db_element_2))

    return DualBasis(elements=dbe_list) # db


def d2_g2_mapping(dim):
    """
    Generate the mapping between d2 and g2

    :param dim: linear dimension of the 1-RDM
    :return: the dual basis of the mapping
    :rtype: DualBasis
    """
    krond = np.eye(dim)
    db = DualBasis()
    dbe_list = []

    def g2d2map(p, q, r, s, factor=1):
        """
        Build the dual basis element for a symmetric 2-marginal

        :param p: tensor index
        :param q: tensor index
        :param r: tensor index
        :param s: tensor index
        :param factor: weighting of the element
        :return: the dual basis element
        """
        dbe = DualBasisElement()
        dbe.add_element('ck', (p, r), -1. * krond[q, s] * factor)
        dbe.add_element('ckck', (p, s, r, q), 1.0 * factor)
        dbe.add_element('cckk', (p, q, r, s), 1.0 * factor)
        dbe.dual_scalar = 0

        return dbe

    for p, q, r, s in product(range(dim), repeat=4):
        if p * dim + q <= r * dim + s:
            db_element = g2d2map(p, q, r, s, factor=0.5)
            db_element_2 = g2d2map(r, s, p, q, factor=0.5)
            # db += db_element.join_elements(db_element_2)
            dbe_list.append(db_element.join_elements(db_element_2))

    return DualBasis(elements=dbe_list)  # db


def d2_e2_mapping(dim, measured_tpdm):
    """
    Generate the constraints between the error matrix, d2, and a measured d2.

    :param dim: dimension of the spin-orbital basis
    :param measured_tpdm:  a 4-tensor of the measured 2-p
    :return:
    """
    db = DualBasis()
    for p, q, r, s in product(range(dim), repeat=4):
        if p * dim + q >= r * dim + s:
            dbe = DualBasisElement()
            # two elements of d2
            dbe.add_element('cckk', (p, q, r, s), 0.5)
            dbe.add_element('cckk', (r, s, p, q), 0.5)

            # add four elements of the error matrix
            dbe.add_element('cckk_me', (p * dim + q + dim**2, r * dim + s), 0.25)
            dbe.add_element('cckk_me', (r * dim + s + dim**2, p * dim + q), 0.25)
            dbe.add_element('cckk_me', (p * dim + q, r * dim + s + dim**2), 0.25)
            dbe.add_element('cckk_me', (r * dim + s, p * dim + q + dim**2), 0.25)

            dbe.dual_scalar = measured_tpdm[p, q, r, s].real
            dbe.simplify()

            # construct the dual basis element for constraining the [0, 0] orthant to be identity matrix
            dbe_idenity = DualBasisElement()
            if p * dim + q == r * dim + s:
                dbe_idenity.add_element('cckk_me', (p * dim + q, r * dim + s), 1.0)
                dbe_idenity.dual_scalar = 1.0
            else:
                # will a symmetric constraint provide variational freedom?
                dbe_idenity.add_element('cckk_me', (p * dim + q, r * dim + s), 0.5)
                dbe_idenity.add_element('cckk_me', (r * dim + s, p * dim + q), 0.5)
                dbe_idenity.dual_scalar = 0.0

            db += dbe
            db += dbe_idenity

    return db


def d2_to_t1_matrix_antisym(dim):
    """
    Generate the dual basis elements for mapping d2 and d1 to the T1-matrix

    The T1 condition is the sum of three-marginals

    T1 = <p^ q^ r^ i j k>  i j k p^ q^ r^>

    in such a fashion that any triples component cancels out. T1 will be
    represented over antisymmeterized basis functions to save a significant amount
    of space

    :param dim: spin-orbital basis dimension
    :return:
    """
    db = []  # DualBasis()
    for p, q, r, i, j, k in product(range(dim), repeat=6):
        if (p * dim**2 + q * dim + r <= i * dim**2 + j * dim + k and p < q < r and i < j < k):
            print(p, q, r, i, j, k)
            dbe = DualBasisElement()
            if np.isclose(p * dim**2 + q * dim + r, i * dim**2 + j * dim + k):
                for pp, qq, rr, cparity in _three_parity(p, q, r):
                    for ii, jj, kk, rparity in _three_parity(i, j, k):
                        # diagonal term should be treated once
                        dbe.dual_scalar -= t1_dual_scalar(pp, qq, rr, ii, jj, kk) * cparity * rparity / 6
                        for element in t1_opdm_component(pp, qq, rr, ii, jj, kk):
                            # dbe.join_elements(element)
                            dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], cparity * rparity * element.primal_coeffs[0] / 6)
                        for element in t1_tpdm_component(pp, qq, rr, ii, jj, kk):
                            # dbe.join_elements(element)
                            dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], cparity * rparity * element.primal_coeffs[0] / 6)
                dbe.add_element('t1', (p, q, r, i, j, k), -1.0)
            else:
                for pp, qq, rr, cparity in _three_parity(p, q, r):
                    for ii, jj, kk, rparity in _three_parity(i, j, k):
                        dbe.dual_scalar -= t1_dual_scalar(pp, qq, rr, ii, jj, kk) * 0.5 / 6
                        for element in t1_opdm_component(pp, qq, rr, ii, jj, kk):
                            dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], cparity * rparity * element.primal_coeffs[0] * 0.5 / 6)
                        for element in t1_tpdm_component(pp, qq, rr, ii, jj, kk):
                            dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], cparity * rparity * element.primal_coeffs[0] * 0.5 / 6)
                dbe.add_element('t1', (p, q, r, i, j, k), -0.5)

                for pp, qq, rr, cparity in _three_parity(i, j, k):
                    for ii, jj, kk, rparity in _three_parity(p, q, r):
                        dbe.dual_scalar -= t1_dual_scalar(pp, qq, rr, ii, jj, kk) * 0.5 / 6
                        for element in t1_opdm_component(pp, qq, rr, ii, jj, kk):
                            # dbe.join_elements(element)
                            dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], cparity * rparity * element.primal_coeffs[0] * 0.5 / 6)
                        for element in t1_tpdm_component(pp, qq, rr, ii, jj, kk):
                            # dbe.join_elements(element)
                            dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], cparity * rparity * element.primal_coeffs[0] * 0.5 / 6)
                dbe.add_element('t1', (i, j, k, p, q, r), -0.5)

            dbe.simplify()
            db.append(dbe)
    return DualBasis(elements=db)


def d2_to_t1_matrix(dim):
    """
    Generate the dual basis elements for mapping d2 and d1 to the T1-matrix

    The T1 condition is the sum of three-marginals

    T1 = <p^ q^ r^ i j k>  i j k p^ q^ r^>

    in such a fashion that any triples component cancels out. T1 will be
    represented over antisymmeterized basis functions to save a significant amount
    of space

    :param dim: spin-orbital basis dimension
    :return:
    """
    db = []  # DualBasis()
    for p, q, r, i, j, k in product(range(dim), repeat=6):
        if (p * dim**2 + q * dim + r <= i * dim**2 + j * dim + k and p != q and q != r and p != r and i != j and j != k and i != k):
            print(p, q, r, i, j, k)
            dbe = DualBasisElement()
            if np.isclose(p * dim**2 + q * dim + r, i * dim**2 + j * dim + k):
                # diagonal term should be treated once
                dbe.dual_scalar -= t1_dual_scalar(p, q, r, i, j, k)
                for element in t1_opdm_component(p, q, r, i, j, k):
                    # dbe.join_elements(element)
                    dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], element.primal_coeffs[0])
                for element in t1_tpdm_component(p, q, r, i, j, k):
                    # dbe.join_elements(element)
                    dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], element.primal_coeffs[0])
                dbe.add_element('t1', (p, q, r, i, j, k), -1.0)
            else:
                dbe.dual_scalar -= t1_dual_scalar(p, q, r, i, j, k) * 0.5
                for element in t1_opdm_component(p, q, r, i, j, k):
                    # dbe.join_elements(element)
                    dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], element.primal_coeffs[0] * 0.5)
                for element in t1_tpdm_component(p, q, r, i, j, k):
                    # dbe.join_elements(element)
                    dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], element.primal_coeffs[0] * 0.5)

                dbe.dual_scalar -= t1_dual_scalar(i, j, k, p, q, r) * 0.5
                for element in t1_opdm_component(i, j, k, p, q, r):
                    # dbe.join_elements(element)
                    dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], element.primal_coeffs[0] * 0.5)
                for element in t1_tpdm_component(i, j, k, p, q, r):
                    # dbe.join_elements(element)
                    dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], element.primal_coeffs[0] * 0.5)

                # This is the weirdest part right here!
                # we reference everything else in tensor ordering but then this is
                # put into a weird matrix order.  The way it works is that it
                # just groups them
                dbe.add_element('t1', (p, q, r, i, j, k), -0.5)
                dbe.add_element('t1', (i, j, k, p, q, r), -0.5)

            dbe.simplify()
            db.append(dbe)
    return DualBasis(elements=db)


def d2_to_t1_from_iterator(dim):
    """
    Generate T1 from the iteratively generated dbe elements

    :param dim:
    :return:
    """
    db = []
    # NOTE: Figure out why join_elements is not working
    for p, q, r, i, j, k in product(range(dim), repeat=6):
        if p != q and q != r and p != r and i != j and j != k and i != k:
            print(p, q, r, i, j, k)
            dbe = DualBasisElement()
            dbe.dual_scalar -= t1_dual_scalar(p, q, r, i, j, k)
            for element in t1_opdm_component(p, q, r, i, j, k):
                # dbe.join_elements(element)
                # print(element.primal_tensors_names, element.primal_elements, element.primal_coeffs)
                dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], element.primal_coeffs[0])
            for element in t1_tpdm_component(p, q, r, i, j, k):
                # dbe.join_elements(element)
                # print(element.primal_tensors_names, element.primal_elements, element.primal_coeffs)
                dbe.add_element(element.primal_tensors_names[0], element.primal_elements[0], element.primal_coeffs[0])
            dbe.add_element('t1', (p, q, r, i, j, k), -1.0)

            db.append(dbe)

    return DualBasis(elements=db)


def d2_to_t1(dim):
    """
    Generate the dual basis elements for mapping d2 and d1 to the T1-matrix

    The T1 condition is the sum of three-marginals

    T1 = <p^ q^ r^ i j k>  i j k p^ q^ r^>

    in such a fashion that any triples component cancels out. T1 will be
    represented over antisymmeterized basis functions to save a significant amount
    of space

    :param dim: spin-orbital basis dimension
    :return:
    """
    db = []  # DualBasis()
    for p, q, r, i, j, k in product(range(dim), repeat=6):
        # if (p * dim**2 + q * dim + r <= i * dim**2 + j * dim + k):
        print(p, q, r, i, j, k)
        dbe = DualBasisElement()
        dbe.dual_scalar -= ((-1.0) * kdelta(i, p) * kdelta(j, q) * kdelta(k, r) +
                            ( 1.0) * kdelta(i, p) * kdelta(j, r) * kdelta(k, q) +
                            ( 1.0) * kdelta(i, q) * kdelta(j, p) * kdelta(k, r) +
                            (-1.0) * kdelta(i, q) * kdelta(j, r) * kdelta(k, p) +
                            (-1.0) * kdelta(i, r) * kdelta(j, p) * kdelta(k, q) +
                            ( 1.0) * kdelta(i, r) * kdelta(j, q) * kdelta(k, p))
        dbe.add_element('ck', (r, k), ( 1.0) * kdelta(i, p) * kdelta(j, q))
        dbe.add_element('ck', (q, k), (-1.0) * kdelta(i, p) * kdelta(j, r))
        dbe.add_element('ck', (r, j), (-1.0) * kdelta(i, p) * kdelta(k, q))
        dbe.add_element('ck', (q, j), ( 1.0) * kdelta(i, p) * kdelta(k, r))
        dbe.add_element('ck', (r, k), (-1.0) * kdelta(i, q) * kdelta(j, p))
        dbe.add_element('ck', (p, k), ( 1.0) * kdelta(i, q) * kdelta(j, r))
        dbe.add_element('ck', (r, j), ( 1.0) * kdelta(i, q) * kdelta(k, p))
        dbe.add_element('ck', (p, j), (-1.0) * kdelta(i, q) * kdelta(k, r))
        dbe.add_element('ck', (q, k), ( 1.0) * kdelta(i, r) * kdelta(j, p))
        dbe.add_element('ck', (p, k), (-1.0) * kdelta(i, r) * kdelta(j, q))
        dbe.add_element('ck', (q, j), (-1.0) * kdelta(i, r) * kdelta(k, p))
        dbe.add_element('ck', (p, j), ( 1.0) * kdelta(i, r) * kdelta(k, q))
        dbe.add_element('ck', (r, i), ( 1.0) * kdelta(j, p) * kdelta(k, q))
        dbe.add_element('ck', (q, i), (-1.0) * kdelta(j, p) * kdelta(k, r))
        dbe.add_element('ck', (r, i), (-1.0) * kdelta(j, q) * kdelta(k, p))
        dbe.add_element('ck', (p, i), ( 1.0) * kdelta(j, q) * kdelta(k, r))
        dbe.add_element('ck', (q, i), ( 1.0) * kdelta(j, r) * kdelta(k, p))
        dbe.add_element('ck', (p, i), (-1.0) * kdelta(j, r) * kdelta(k, q))
        dbe.add_element('cckk', (q, r, j, k), ( 1.0) * kdelta(i, p))
        dbe.add_element('cckk', (p, r, j, k), (-1.0) * kdelta(i, q))
        dbe.add_element('cckk', (p, q, j, k), ( 1.0) * kdelta(i, r))
        dbe.add_element('cckk', (q, r, i, k), (-1.0) * kdelta(j, p))
        dbe.add_element('cckk', (p, r, i, k), ( 1.0) * kdelta(j, q))
        dbe.add_element('cckk', (p, q, i, k), (-1.0) * kdelta(j, r))
        dbe.add_element('cckk', (q, r, i, j), ( 1.0) * kdelta(k, p))
        dbe.add_element('cckk', (p, r, i, j), (-1.0) * kdelta(k, q))
        dbe.add_element('cckk', (p, q, i, j), ( 1.0) * kdelta(k, r))
        dbe.add_element('t1', (p, q, r, i, j, k), -1.0)
        dbe.simplify()
        db.append(dbe)
    return DualBasis(elements=db)


def t1_dual_scalar(p, q, r, i, j, k):
    term = ((-1.00000) * kdelta(i, p) * kdelta(j, q) * kdelta(k, r) +
            ( 1.00000) * kdelta(i, p) * kdelta(j, r) * kdelta(k, q) +
            ( 1.00000) * kdelta(i, q) * kdelta(j, p) * kdelta(k, r) +
            (-1.00000) * kdelta(i, q) * kdelta(j, r) * kdelta(k, p) +
            (-1.00000) * kdelta(i, r) * kdelta(j, p) * kdelta(k, q) +
            ( 1.00000) * kdelta(i, r) * kdelta(j, q) * kdelta(k, p))
    return term


class DualElementStructGenerator:
    """Class for representing a single component of a dual basis element

    For example, the T1 map has 1-RDM mapping terms that look like this:

    (1.00000) * kdelta(i, p) * kdelta(j, q) * opdm[r, k] +
    (-1.00000) * kdelta(i, p) * kdelta(j, r) * opdm[q, k] +
    (-1.00000) * kdelta(i, p) * kdelta(k, q) * opdm[r, j] +
    (1.00000) * kdelta(i, p) * kdelta(k, r) * opdm[q, j] +
    (-1.00000) * kdelta(i, q) * kdelta(j, p) * opdm[r, k] +
    (1.00000) * kdelta(i, q) * kdelta(j, r) * opdm[p, k] +
    (1.00000) * kdelta(i, q) * kdelta(k, p) * opdm[r, j] +
    (-1.00000) * kdelta(i, q) * kdelta(k, r) * opdm[p, j] +
    (1.00000) * kdelta(i, r) * kdelta(j, p) * opdm[q, k] +
    (-1.00000) * kdelta(i, r) * kdelta(j, q) * opdm[p, k] +
    (-1.00000) * kdelta(i, r) * kdelta(k, p) * opdm[q, j] +
    (1.00000) * kdelta(i, r) * kdelta(k, q) * opdm[p, j] +
    (1.00000) * kdelta(j, p) * kdelta(k, q) * opdm[r, i] +
    (-1.00000) * kdelta(j, p) * kdelta(k, r) * opdm[q, i] +
    (-1.00000) * kdelta(j, q) * kdelta(k, p) * opdm[r, i] +
    (1.00000) * kdelta(j, q) * kdelta(k, r) * opdm[p, i] +
    (1.00000) * kdelta(j, r) * kdelta(k, p) * opdm[q, i] +
    (-1.00000) * kdelta(j, r) * kdelta(k, q) * opdm[p, i]

    we need infrastructure to store the coefficient, the elements in the delta functions
    and the marginal.  This is almost looks like a collection of elements of a
    dual basis but is inefficient.  It is expensive to simplify dual basis elements
    so we want to front load the computation by only providing non-zero elements
    of the dual basis element.  Therefore, we want to make a generator that takes
    the indices p, q, r, i, j, k and returns the one-RDM components that are non
    zero in a generator.
    """
    def __init__(self, coeff=None, deltas=None, tensor_name=None, tensor_element=None):
        self.coeff = coeff
        self.deltas = deltas
        self.tensor_name = tensor_name
        self.tensor_element = tensor_element


def t1_opdm_component(p, q, r, i, j, k):
    """
    Iterate through one-RDM mapping components of T1 map

    :param Int p: Index for T1 matrix
    :param Int q: Index for T1 matrix
    :param Int r: Index for T1 matrix
    :param Int i: Index for T1 matrix
    :param Int j: Index for T1 matrix
    :param Int k: Index for T1 matrix
    :return: yield non-zero elements
    """
    terms = [DualElementStructGenerator( 1.0, [(i, p), (j, q)], 'ck', (r, k)),
             DualElementStructGenerator(-1.0, [(i, p), (j, r)], 'ck', (q, k)),
             DualElementStructGenerator(-1.0, [(i, p), (k, q)], 'ck', (r, j)),
             DualElementStructGenerator( 1.0, [(i, p), (k, r)], 'ck', (q, j)),
             DualElementStructGenerator(-1.0, [(i, q), (j, p)], 'ck', (r, k)),
             DualElementStructGenerator( 1.0, [(i, q), (j, r)], 'ck', (p, k)),
             DualElementStructGenerator( 1.0, [(i, q), (k, p)], 'ck', (r, j)),
             DualElementStructGenerator(-1.0, [(i, q), (k, r)], 'ck', (p, j)),
             DualElementStructGenerator( 1.0, [(i, r), (j, p)], 'ck', (q, k)),
             DualElementStructGenerator(-1.0, [(i, r), (j, q)], 'ck', (p, k)),
             DualElementStructGenerator(-1.0, [(i, r), (k, p)], 'ck', (q, j)),
             DualElementStructGenerator( 1.0, [(i, r), (k, q)], 'ck', (p, j)),
             DualElementStructGenerator( 1.0, [(j, p), (k, q)], 'ck', (r, i)),
             DualElementStructGenerator(-1.0, [(j, p), (k, r)], 'ck', (q, i)),
             DualElementStructGenerator(-1.0, [(j, q), (k, p)], 'ck', (r, i)),
             DualElementStructGenerator( 1.0, [(j, q), (k, r)], 'ck', (p, i)),
             DualElementStructGenerator( 1.0, [(j, r), (k, p)], 'ck', (q, i)),
             DualElementStructGenerator(-1.0, [(j, r), (k, q)], 'ck', (p, i))]
    # Create the generator that yeilds the next non-zero 1-RDM component
    for desg_term in terms:
        dbe = DualBasisElement()
        delta_val = 1.0
        for krond_pair in desg_term.deltas:
            delta_val *= kdelta(*krond_pair)
        if np.isclose(delta_val, 1.0):
            dbe.add_element(desg_term.tensor_name, desg_term.tensor_element,
                            desg_term.coeff)
            yield dbe


def t1_tpdm_component(p, q, r, i, j, k):
    """
    Iterate through two-RDM mapping components of T1 map

    :param Int p: index for T1 term
    :param Int q: index for T1 term
    :param Int r: index for T1 term
    :param Int i: index for T1 term
    :param Int j: index for T1 term
    :param Int k: index for T1 term
    :return: Generator yielding DualbasisElements
    """
    terms = [DualElementStructGenerator( 1.0, [(i, p)], 'cckk', (q, r, j, k)),
             DualElementStructGenerator(-1.0, [(i, q)], 'cckk', (p, r, j, k)),
             DualElementStructGenerator( 1.0, [(i, r)], 'cckk', (p, q, j, k)),
             DualElementStructGenerator(-1.0, [(j, p)], 'cckk', (q, r, i, k)),
             DualElementStructGenerator( 1.0, [(j, q)], 'cckk', (p, r, i, k)),
             DualElementStructGenerator(-1.0, [(j, r)], 'cckk', (p, q, i, k)),
             DualElementStructGenerator( 1.0, [(k, p)], 'cckk', (q, r, i, j)),
             DualElementStructGenerator(-1.0, [(k, q)], 'cckk', (p, r, i, j)),
             DualElementStructGenerator( 1.0, [(k, r)], 'cckk', (p, q, i, j))]

    # create the generator that yeilds the next non-zero 2-RDM component
    for desg_term in terms:
        dbe = DualBasisElement()
        delta_val = 1.0
        for krond_pair in desg_term.deltas:
            delta_val *= kdelta(*krond_pair)
        if np.isclose(delta_val, 1.0):
            dbe.add_element(desg_term.tensor_name, desg_term.tensor_element,
                            desg_term.coeff)
            yield dbe


def spin_orbital_linear_constraints(dim, N, constraint_list):
    """
    Genrate the dual basis for the v2-RDM program

    :param dim: rank of spinless fermion basis
    :param N: Total number of electrons
    :param constraint_list:  List of strings indicating which constraints to make
    :return:  Dual basis for the constraint program
    :rtype: DualBasis
    """

    dual_basis = DualBasis()
    if 'cckk' in constraint_list:
        print("d2 constraints")
        # trace constraint on D2
        dual_basis += trace_constraint(dim, N * (N - 1))

        # antisymmetry constraint
        print("antisymmetry constraint")
        dual_basis += antisymmetry_constraints(dim)

    if 'ck' in constraint_list:
        print("opdm constraints")
        dual_basis += d2_d1_mapping(dim, N - 1)
        dual_basis += d1_q1_mapping(dim)

    # d2 -> q2
    if 'kkcc' in constraint_list:
        print('tqdm constraints')
        dual_basis += d2_q2_mapping(dim)

    # d2 -> g2
    if "ckck" in constraint_list:
        print('phdm constraints')
        dual_basis += d2_g2_mapping(dim)

    # d2, d1 -> T1
    if "t1" in constraint_list:
        # this uses an antisymmeterized form of the T1 matrix since it's really
        # not feasible to store r^{6} elements
        print('t1 constrinat')
        dual_basis += d2_to_t1_matrix_antisym(dim)

    return dual_basis
