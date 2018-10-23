"""The purpose of this module is to provide routines to calculate expected
values of marginals useful in benchmarking quantum algorithms.  For example,
Cumulant decompositions, Expected SZ, expected S2, Expected number operator

SZ = 0.5 * \sum_{i = 1}^{n} n_{i \alpha} - n_{i, \beta}
S^{2} = S_ S+ + Sz( SZ + 1)
S_ = \sum_{i=1}^{n} a_{n, \beta}^{\dagger}a_{n, \alpha}
S_ += \sum_{i=1}^{n} a_{n, \alpha}^{\dagger}a_{n, \beta}
N = \sum_{i = 1}^{n} n_{i \alpha} + n_{i, \beta}
"""
from itertools import product
import numpy as np


def sz_expected(opdm):
    """Calculate the Sz expected value from a one-particle marginal

    SZ = 0.5 * \sum_{i = 1}^{n} n_{i \alpha} - n_{i, \beta}

    assume even indices correspond to alpha-spin-orbitals. odd indices
    correspond to beta spin orbitals

    :param opdm: one-particle marginal (1, 1)-tensor
    """
    sz_eval = 0.0
    for i in range(opdm.shape[0] // 2):
        sz_eval += opdm[2 * i, 2 * i] - opdm[2 * i + 1, 2 * i + 1]

    return 0.5 * sz_eval


def s2_expected(tpdm, opdm):
    """
    Calculate the S^{2} operator by evaluating directly from 1- and 2-RDM

    Note: alpha takes even
    :param tpdm: two-particle RDM in the spin-oribtal basis
    :param opdm: one-particle RDM in the spin-orbital basis
    :return:
    """
    sminussplus = 0.0
    krond = np.eye(opdm.shape[0])
    spatial_dim = int(opdm.shape[0] / 2.0)
    for i, j in product(range(spatial_dim), repeat=2):
        sminussplus += opdm[2 * i + 1, 2 * j + 1] * krond[2 * i + 1, 2 * j + 1]
        sminussplus -= tpdm[2 * j, 2 * i + 1, 2 * i, 2 * j + 1]

    szsquared = 0.0
    for p, q in product(range(spatial_dim), repeat=2):
        szsquared += 0.25 * opdm[2 * p, 2 * q] * krond[2 * p, 2 * q]
        szsquared += 0.25 * opdm[2 * p + 1, 2 * q + 1] * krond[2 * p + 1, 2 * q + 1]

        szsquared += 0.25 * tpdm[2 * p, 2 * q, 2 * p, 2 * q]
        szsquared += 0.25 * tpdm[2 * p + 1, 2 * q + 1, 2 * p + 1, 2 * q + 1]

        szsquared -= 0.25 * tpdm[2 * p, 2 * q + 1, 2 * p, 2 * q + 1]
        szsquared -= 0.25 * tpdm[2 * p + 1, 2 * q, 2 * p + 1, 2 * q]

    sz = sz_expected(opdm)

    return sminussplus + szsquared + sz


def number_expectation(opdm):
    """Calculate the number of particles represented in the 1-marginal

    :param opdm: one-particle marginal as a (1, 1)-tensor
    """
    return np.trace(opdm)


def number_alpha_expectation(opdm):
    """Calculate the number of alpha particles in the one-marginal

    :param opdm: one-particle marginal as a (1, 1)-tensor
    """
    return np.trace(opdm[::2, ::2])


def number_beta_expectation(opdm):
    """Calculate the number of beta particles in the one-marginal

    :param opdm: one-particle marginal as a (1, 1)-tensor
    """
    return np.trace(opdm[1::2, 1::2])
