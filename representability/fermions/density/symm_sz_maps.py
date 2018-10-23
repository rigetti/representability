"""
Map marginals between each one another by Fermionic anticommutaiton relations

A collection of methods that accept a RDM and map it to another RDM
"""
import numpy as np
from itertools import product


def map_d1_q1(opdm):
    """
    map the one-particle density matrix to the one-hole density matrix

    :param opdm: rank-2 tensor representing the 1-RDM
    :returns: rank-2 tensor representing the one-hole-RDM
    :rtype: ndarray
    """
    m = opdm.shape[0]
    I = np.eye(m)
    return I - opdm


def map_d2_d1(tpdm, normalization):
    """
    map the two-particle density matrix to the one-particle density matrix

    This is a contraction operation

    :param tpdm: rank-4 tensor representing the 2-RDM
    :param normalization: normalization constant for mapping
    :returns: rank-2 tensor representing the 1-RDM
    :rtype: ndarray
    """
    sm_dim = tpdm.shape[0]
    opdm = np.zeros((sm_dim, sm_dim), dtype=complex)
    for p, q, in product(range(sm_dim), repeat=2):
        term = 0
        for r in range(sm_dim):
            term += tpdm[p, r, q, r]
        opdm[p, q] = term/normalization

    return opdm


def map_d2_q2(tpdm, opdm):
    """
    map the two-particle density matrix to the two-hole density matrix

    :param tpdm: rank-4 tensor representing the 2-RDM
    :param opdm: rank-2 tensor representing the 1-RDM
    :returns: rank-4 tensor representing the 2-H-RDM
    :rtype: ndarray
    """
    sm_dim = opdm.shape[0]
    krond = np.eye(sm_dim)
    tqdm = np.zeros((sm_dim, sm_dim, sm_dim, sm_dim), dtype=complex)
    for p, q, r, s in product(range(sm_dim), repeat=4):
        term1 = opdm[p, r]*krond[q, s] + opdm[q, s]*krond[p, r]
        term2 = -1*(opdm[p, s]*krond[r, q] + opdm[q, r]*krond[s, p])
        term3 = krond[s, p]*krond[r, q] - krond[q, s]*krond[r, p]
        term4 = tqdm[r, s, p, q]
        tqdm[p, q, r, s] = tpdm[p, q, r, s] - term1 - term2 - term3

    return tqdm


def map_d2_q2_ab(tpdm, opdm_a, opdm_b):
    """
    map the two-particle density matrix to the two-hole density matrix

    :param tpdm: rank-4 tensor representing the 2-RDM
    :param opdm_a: rank-2 tensor representing the 1-RDM-alpha block
    :param opdm_b: rank-2 tensor representing the 1-RDM-beta block
    :returns: rank-4 tensor representing the 2-H-RDM-ab or ba
    :rtype: ndarray
    """
    sm_dim = opdm_a.shape[0]
    krond = np.eye(sm_dim)
    tqdm = np.zeros_like(tpdm)
    for p, q, r, s in product(range(sm_dim), repeat=4):
        term1 = opdm_a[p, r]*krond[q, s] + opdm_b[q, s]*krond[p, r]
        term2 = -krond[p, r]*krond[q, s]
        tqdm[r, s, p ,q] = tpdm[p, q, r, s] - term1 - term2
        assert np.isclose(tpdm[p, q, r, s], term1 + term2 + tqdm[r, s, p, q])

    return tqdm


def map_d2_g2(tpdm, opdm):
    """
    map the two-particle density matrix to the particle-hold density matrix

    :param tpdm: rank-4 tensor representing the 2-RDM
    :param opdm: rank-2 tensor representing the 1-RDM
    :returns: rank-4 tensor representing the P-H-DM
    :rtype: ndarray
    """
    sm_dim = opdm.shape[0]
    krond = np.eye(sm_dim)
    tgdm = np.zeros_like(tpdm)
    for p, q, r, s in product(range(sm_dim), repeat=4):
        term1 = opdm[p, r]*krond[q, s]
        term2 = -1*tgdm[p, s, r, q]
        tgdm[p, s, r, q] = term1 - tpdm[p, q, r, s]
    return tgdm


def map_d2_g2_sz(tpdm, opdm, g2_block='aaaa'):
    """
    Map elements of D2 to elements of G2
    """
    sm_dim = opdm.shape[0]
    krond = np.eye(sm_dim)
    tgdm = np.zeros((sm_dim, sm_dim, sm_dim, sm_dim), dtype=complex)
    if g2_block == 'aaaa' or g2_block == 'bbbb':
        for p, q, r, s in product(range(sm_dim), repeat=4):
            term1 = opdm[p, r]*krond[q, s]
            term2 = -1*tpdm[p, s, r, q]
            # print tgdm[p, q, r, s], term1 + term2
            # assert np.isclose(tgdm[p, q, r, s], term1 + term2)
            tgdm[p, q, r, s] = term1 + term2

    elif g2_block == 'bbaa' or g2_block == 'aabb':
        for p, q, r, s in product(range(sm_dim), repeat=4):
            term1 = tpdm[p, s, q, r]
            # print tgdm[p, q, r, s,], term1
            # assert np.isclose(tgdm[p, q, r, s], term1)
            tgdm[p, q, r, s] = term1
    elif g2_block == 'abab' or g2_block == 'baba':
        for p, q, r, s in product(range(sm_dim), repeat=4):
            term1 = opdm[p, r]*krond[q, s]
            # just in case the user sends a rank-4 tensor as a rank-2 tensor
            if tpdm.ndim == 2:
                term2 = -1*tpdm[p*sm_dim + s, r*sm_dim + q]
            else:
                term2 = -1*tpdm[p, s, r, q]
            # assert np.isclose(tgdm[p, q, r, s], term1 + term2)
            tgdm[p, q, r, s] = term1 + term2

    return tgdm
