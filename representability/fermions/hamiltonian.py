"""
A module for constructing Hamiltonians for marginal reconstruction and variational 2-RDM theory.

functionality to transform molecular integrals into the appropriate Tensor objects
"""
from itertools import product
import numpy as np
from scipy.linalg import block_diag
from representability.fermions.basis_utils import geminal_spin_basis
from representability.tensor import Tensor


def spin_orbital_interaction_tensor(two_body_int, one_body_int):
    """
    Construct the cost operator

    :param two_body_int: two-body integrals in spin-orbital basis
    :param one_body_int: one-body integral in spin-orbital basis
    """
    opdm_interaction_tensor = Tensor(one_body_int, name='ck')
    tpdm_interaction_tensor = Tensor(two_body_int, name='cckk')

    return opdm_interaction_tensor, tpdm_interaction_tensor


def spin_adapted_interaction_tensor(two_body_int, one_body_int):
    """
    Construct the cost operator in symmetric and antisymmetric basis

    The spin-orbital integrals are in the spin-less fermion basis.
    Spin-full fermions are index by even/odd

    :param two_body_int:
    :param one_body_int:
    :return:
    """
    sp_dim = int(one_body_int.shape[0] / 2)
    one_body_spatial_int = np.zeros((sp_dim, sp_dim), dtype=float)
    even_set = one_body_int[::2, ::2].copy()
    for p, q in product(range(sp_dim), repeat=2):
        one_body_spatial_int[p, q] = one_body_int[2 * p, 2 * q]

    assert np.allclose(even_set, one_body_spatial_int)
    opdm_a_interaction = Tensor(one_body_spatial_int, name='ck_a')
    opdm_b_interaction = Tensor(one_body_spatial_int, name='ck_b')

    aa_dim = int(sp_dim * (sp_dim - 1) / 2)
    ab_dim = int(sp_dim**2)
    v2aa = np.zeros((aa_dim, aa_dim))
    v2bb = np.zeros_like(v2aa)
    v2ab = np.zeros((ab_dim, ab_dim))

    b_aa_dict = {}
    b_ab_dict = {}
    cnt, cnt2 = 0, 0
    for i, j in product(range(sp_dim), repeat=2):
        if i < j:
            b_aa_dict[(i, j)] = cnt
            cnt += 1
        b_ab_dict[(i, j)] = cnt2
        cnt2 += 1

    for p, q, r, s in product(range(sp_dim), repeat=4):
        if p < q and r < s:
            # 0.5 still there because antisymmetric basis becomes <ij|kl> -
            # <ij|lk>.  The 0.5 for coulomb interaction counting is still
            # needed to avoid double counting
            v2aa[b_aa_dict[(p, q)], b_aa_dict[(r, s)]] = 0.5 * (two_body_int[2 * p, 2 * q, 2 * r, 2 * s] - two_body_int[2 * p, 2 * q, 2 * s, 2 * r])
            v2bb[b_aa_dict[(p, q)], b_aa_dict[(r, s)]] = 0.5 * (two_body_int[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] - two_body_int[2 * p + 1, 2 * q + 1, 2 * s + 1, 2 * r + 1])

        v2ab[b_ab_dict[(p, q)], b_ab_dict[(r, s)]] = two_body_int[2 * p, 2 * q + 1, 2 * r, 2 * s + 1]

    bas_aa, bas_ab = geminal_spin_basis(sp_dim)
    v2ab = Tensor(v2ab, basis=bas_ab, name='cckk_ab')
    v2bb = Tensor(v2bb, basis=bas_aa, name='cckk_bb')
    v2aa = Tensor(v2aa, basis=bas_aa, name='cckk_aa')

    return opdm_a_interaction, opdm_b_interaction, v2aa, v2bb, v2ab


def spin_adapted_interaction_tensor_rdm_consistent(two_body_int, one_body_int):
    """
    Construct the cost operator in symmetric and antisymmetric basis

    The spin-orbital integrals are in the spin-less fermion basis.
    Spin-full fermions are index by even/odd

    :param two_body_int:
    :param one_body_int:
    :return:
    """
    sp_dim = int(one_body_int.shape[0] / 2)
    one_body_spatial_int = np.zeros((sp_dim, sp_dim), dtype=float)
    even_set = one_body_int[::2, ::2].copy()
    for p, q in product(range(sp_dim), repeat=2):
        one_body_spatial_int[p, q] = one_body_int[2 * p, 2 * q]

    assert np.allclose(even_set, one_body_spatial_int)
    opdm_a_interaction = Tensor(one_body_spatial_int, name='ck_a')
    opdm_b_interaction = Tensor(one_body_spatial_int, name='ck_b')

    aa_dim = int(sp_dim * (sp_dim - 1) / 2)
    ab_dim = int(sp_dim**2)
    v2aa = np.zeros((aa_dim, aa_dim))
    v2bb = np.zeros_like(v2aa)
    v2ab = np.zeros((ab_dim, ab_dim))

    b_aa_dict = {}
    b_ab_dict = {}
    cnt, cnt2 = 0, 0
    for i, j in product(range(sp_dim), repeat=2):
        if i < j:
            b_aa_dict[(i, j)] = cnt
            cnt += 1
        b_ab_dict[(i, j)] = cnt2
        cnt2 += 1

    for p, q, r, s in product(range(sp_dim), repeat=4):
        if p < q and r < s:
            # 0.5 still there because antisymmetric basis becomes <ij|kl> -
            # <ij|lk>.  The 0.5 for coulomb interaction counting is still
            # needed to avoid double counting
            v2aa[b_aa_dict[(p, q)], b_aa_dict[(r, s)]] = 0.5 * (two_body_int[2 * p, 2 * q, 2 * r, 2 * s] -
                                                                two_body_int[2 * p, 2 * q, 2 * s, 2 * r] -
                                                                two_body_int[2 * q, 2 * p, 2 * r, 2 * s] +
                                                                two_body_int[2 * q, 2 * p, 2 * s, 2 * r])
            v2bb[b_aa_dict[(p, q)], b_aa_dict[(r, s)]] = 0.5 * (two_body_int[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] -
                                                                two_body_int[2 * p + 1, 2 * q + 1, 2 * s + 1, 2 * r + 1] -
                                                                two_body_int[2 * q + 1, 2 * p + 1, 2 * r + 1, 2 * s + 1] +
                                                                two_body_int[2 * q + 1, 2 * p + 1, 2 * s + 1, 2 * r + 1])

        v2ab[b_ab_dict[(p, q)], b_ab_dict[(r, s)]] = two_body_int[2 * p, 2 * q + 1, 2 * r, 2 * s + 1]

    bas_aa, bas_ab = geminal_spin_basis(sp_dim)
    v2ab = Tensor(v2ab, basis=bas_ab, name='cckk_ab')
    v2bb = Tensor(v2bb, basis=bas_aa, name='cckk_bb')
    v2aa = Tensor(v2aa, basis=bas_aa, name='cckk_aa')

    return opdm_a_interaction, opdm_b_interaction, v2aa, v2bb, v2ab


def spin_orbital_marginal_norm_min(dim, tensor_name='ME', basis=None):
    """
    Construct the cost operator as the trace over free variables

    quadrant indexing
    [0, 0] | [0, 1]
    ---------------
    [1, 0] | [1, 1]


    I | E
    -----
    E | F

    Example:

    Mat =
        [ 0,  1,  2,  3,|  4,  5,  6,  7]
        [ 8,  9, 10, 11,| 12, 13, 14, 15]
        [16, 17, 18, 19,| 20, 21, 22, 23]
        [24, 25, 26, 27,| 28, 29, 30, 31]
        ---------------------------------
        [32, 33, 34, 35,| 36, 37, 38, 39]
        [40, 41, 42, 43,| 44, 45, 46, 47]
        [48, 49, 50, 51,| 52, 53, 54, 55]
        [56, 57, 58, 59,| 60, 61, 62, 63]


    M = 2
    for p, q in product(range(M), repeat=2):
        Mat[p*M + q + 1 * M**2, p*M + q + 1 * M**2] = 1.0

    :param Int dim: 2 * dim is the size of the super-block
    :param String tensor_name: name to index the tensor by
    :param Bijection basis: Default None. basis for off-diagonals of superblock
    """

    zero_block = np.zeros((dim, dim))
    eye_block = np.eye(dim)
    cost_tensor = block_diag(zero_block, eye_block)
    cost = Tensor(cost_tensor, basis=basis, name=tensor_name)

    return cost
