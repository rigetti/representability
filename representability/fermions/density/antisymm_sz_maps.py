"""
Map marginals between each one another by Fermionic anticommutaiton relations

A collection of methods that accept a RDM and map it to another RDM
"""
import numpy as np
from itertools import product


def map_d2anti_d1_sz(tpdm, normalization, bas, m_dim):
    """
    construct the opdm from the tpdm and the basis
    """
    test_opdm = np.zeros((m_dim, m_dim), dtype=complex)
    for i, j in product(range(opdm.shape[0]), repeat=2):
        for r in range(opdm.shape[0]):
            if i != r and j != r:
                top_gem = tuple(sorted([i, r]))
                bot_gem = tuple(sorted([j, r]))
                parity = (-1)**(r < i) * (-1)**(r < j)
                test_opdm[i, j] += tpdm[bas[top_gem], bas[bot_gem]] * 0.5 * parity
    test_opdm /= normalization
    return test_opdm


def map_d2symm_d1_sz(tpdm, normalization, bas, m_dim):
    test_opdm = np.zeros((m_dim, m_dim), dtype=complex)
    for i, j in product(range(m_dim), repeat=2):
        for r in range(m_dim):
            test_opdm[i, j] += tpdm[i * m_dim + r, j * m_dim + r]
    test_opdm /= normalization
    return test_opdm


def get_sz_spin_adapted(measured_tpdm):
    """
    Take a spin-orbital 4-tensor 2-RDM and map to the SZ spin adapted version return aa, bb, and ab matrices

    :param measured_tpdm: spin-orbital 2-RDM 4-tensor
    :return: 2-RDM matrices for aa, bb, and ab
    """
    if np.ndim(measured_tpdm) != 4:
        raise TypeError("measured_tpdm must be a 4-tensor")

    dim = measured_tpdm.shape[0]  # spin-orbital basis rank
    # antisymmetric basis dimension
    aa_dim = int((dim / 2) * (dim / 2 - 1) / 2)
    ab_dim = int((dim / 2) ** 2)
    d2_aa = np.zeros((aa_dim, aa_dim))
    d2_bb = np.zeros_like(d2_aa)
    d2_ab = np.zeros((ab_dim, ab_dim))

    # build basis look up table
    bas_aa = {}
    bas_ab = {}
    cnt_aa = 0
    cnt_ab = 0
    # iterate over spatial orbital indices
    for p, q in product(range(dim // 2), repeat=2):
        if q > p:
            bas_aa[(p, q)] = cnt_aa
            cnt_aa += 1
        bas_ab[(p, q)] = cnt_ab
        cnt_ab += 1

    for i, j, k, l in product(range(dim // 2), repeat=4):  # iterate over spatial indices
        d2_ab[bas_ab[(i, j)], bas_ab[(k, l)]] = measured_tpdm[2 * i, 2 * j + 1, 2 * k, 2 * l + 1].real

        if i < j and k < l:
            d2_aa[bas_aa[(i, j)], bas_aa[(k, l)]] = measured_tpdm[2 * i, 2 * j, 2 * k, 2 * l].real - \
                                                    measured_tpdm[2 * i, 2 * j, 2 * l, 2 * k].real - \
                                                    measured_tpdm[2 * j, 2 * i, 2 * k, 2 * l].real + \
                                                    measured_tpdm[2 * j, 2 * i, 2 * l, 2 * k].real

            d2_bb[bas_aa[(i, j)], bas_aa[(k, l)]] = measured_tpdm[2 * i + 1, 2 * j + 1, 2 * k + 1, 2 * l + 1].real - \
                                                    measured_tpdm[2 * i + 1, 2 * j + 1, 2 * l + 1, 2 * k + 1].real - \
                                                    measured_tpdm[2 * j + 1, 2 * i + 1, 2 * k + 1, 2 * l + 1].real + \
                                                    measured_tpdm[2 * j + 1, 2 * i + 1, 2 * l + 1, 2 * k + 1].real

            d2_aa[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5
            d2_bb[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5

    return d2_aa, d2_bb, d2_ab