"""
Map marginals between each one another by Fermionic anticommutaiton relations

A collection of methods that accept a RDM and map it to another RDM
"""
import numpy as np
from itertools import product


def kdelta(i, j):
    return 1.0 if i == j else 0.0


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
        opdm[p, q] = term/(normalization - 1)

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
        # term4 = tqdm[r, s, p, q]
        tqdm[p, q, r, s] = tpdm[p, q, r, s] - term1 - term2 - term3

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


def map_d2_d1_to_t1(tpdm, opdm):
    """
    Map the 2-RDM and 1-RDM to the T1-matrix

    :param tpdm:
    :param opdm:
    :return:
    """
    dim = opdm.shape[0]
    t1 = np.zeros_like((dim, dim, dim, dim, dim, dim), dtype=tpdm.type)
    for p, q, r, i, j, k in product(range(opdm.shape[1]), repeat=6):
        # Note: we need to reorder the RDM because the construct_t1_term expects an OpenFermion ordered tpdm
        term = construct_t1_term(p, q, r, i, j, k, opdm, np.einsum('ijkl->ijlk', tpdm))
        t1[p, q, r, i, j, k] = term
    return t1


def construct_t1_term(p, q, r, i, j, k, opdm, tpdm):
        """
        Construct the T1 matrix term

        This requires tpdm to be in the openfermion ordering
        tpdm[p, q, r, s] = <p^ q^ r s>

        :param p:
        :param q:
        :param r:
        :param i:
        :param j:
        :param k:
        :param opdm:
        :param tpdm:
        :return:
        """
        term = (
                (-1.00000) * kdelta(i, p) * kdelta(j, q) * kdelta(k, r) +
                (1.00000) * kdelta(i, p) * kdelta(j, r) * kdelta(k, q) +
                (1.00000) * kdelta(i, q) * kdelta(j, p) * kdelta(k, r) +
                (-1.00000) * kdelta(i, q) * kdelta(j, r) * kdelta(k, p) +
                (-1.00000) * kdelta(i, r) * kdelta(j, p) * kdelta(k, q) +
                (1.00000) * kdelta(i, r) * kdelta(j, q) * kdelta(k, p) +
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
                (-1.00000) * kdelta(j, r) * kdelta(k, q) * opdm[p, i] +
                (1.00000) * kdelta(i, p) * tpdm[q, r, j, k] +
                (-1.00000) * kdelta(i, q) * tpdm[p, r, j, k] +
                (1.00000) * kdelta(i, r) * tpdm[p, q, j, k] +
                (-1.00000) * kdelta(j, p) * tpdm[q, r, i, k] +
                (1.00000) * kdelta(j, q) * tpdm[p, r, i, k] +
                (-1.00000) * kdelta(j, r) * tpdm[p, q, i, k] +
                (1.00000) * kdelta(k, p) * tpdm[q, r, i, j] +
                (-1.00000) * kdelta(k, q) * tpdm[p, r, i, j] +
                (1.00000) * kdelta(k, r) * tpdm[p, q, i, j])
        return term

