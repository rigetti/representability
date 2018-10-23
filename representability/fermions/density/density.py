"""
Object for handling the N-qubit density matrix and contractions to the family of marginals
"""
import numpy as np
from itertools import product


class Density(object):

    def __init__(self, rho):
        """
        Abstract Density object

        Object that houses the density matrix and is subclassed for particular marginal
        construction.

        :param rho: density matrix of the n-qubit system or wavefunction of the n-qubit
                    system.
        """
        # convert to a density matrix
        if isinstance(rho, list):
            rho = np.asarray(rho)
        if len(rho.shape) == 1:
            rho = np.reshape(rho, (-1, 1))
            rho = rho.dot(np.conj(rho).T)

        num_qubits = int(np.log2(rho.shape[0]))
        self.num_qubits = num_qubits
        self.rho = rho

    def construct_opdm(self):
        "Abstract opdm construction class"
        raise NotImplemented("construct_opdm needs to be implemented by subclass")

    def construct_ohdm(self):
        "Abstract ohdm construction class"
        raise NotImplemented("construct_ohdm needs to be implemented by subclass")

    def construct_tpdm(self):
        "Abstract tpdm construction class"
        raise NotImplemented("construct_tpdm needs to be implemented by subclass")

    def construct_thdm(self):
        "Abstract thdm construction class"
        raise NotImplemented("construct_thdm needs to be implemented by subclass")

    def construct_phdm(self):
        "Abstract phdm construction class"
        raise NotImplemented("construct_phdm needs to be implemented by subclass")

    def construct_pppdm(self):
        "Abstract pppdm construction class"
        raise NotImplemented("construct_pppdm needs to be implemented by subclass")

    def construct_hhhdm(self):
        "Abstract hhhdm construction class"
        raise NotImplemented("construct_hhhdm needs to be implemented by subclass")

    def construct_pphdm(self):
        "Abstract pphdm construction class"
        raise NotImplemented("construct_pphdm needs to be implemented by subclass")

    def construct_phhdm(self):
        "Abstract phhdm construction class"
        raise NotImplemented("construct_phhdm needs to be implemented by subclass")


def check_rank2_trace(marginal, N):
    """
    Check the trace of a marginal that is rank 2

    :param marginal: rank-2 marginal has two indices.  Trace is sum over 'ii'
    :param N: trace should be equal to this number
    :returns: Boolean True/False
    :rtype: Bool
    """
    if marginal.ndim != 2:
        raise TypeError("You input a non-rank-2 tensor")

    return np.isclose(np.einsum('ii', marginal), N)


def check_rank4_trace(marginal, N):
    """
    Check the trace of a marginal that is rank 4

    :param marginal: rank-4 maginal that has 4 indices
    :param N: trace should be equal to this number
    :returns: Boolean True/False
    :rtype: Bool
    """
    if marginal.ndim != 4:
        raise TypeError("You input a non-rank-4 tensor")

    return np.isclose(np.einsum('ijij', marginal), N)


def check_rank6_trace(marginal, N):
    """
    Check the trace of a marginal that is rank 4

    :param marginal: rank-6 maginal that has 6 indices
    :param N: trace should be equal to this number
    :returns: Boolean True/False
    :rtype: Bool
    """
    if marginal.ndim != 6:
        raise TypeError("You input a non-rank-4 tensor")

    return np.isclose(np.einsum('ijkijk', marginal), N)


def check_antisymmetric_marginal(marginal):
    """
    check if rank-4 marginal is antisymmetric
    """
    m = marginal.shape[0]
    for p, q, r, s in product(range(m), repeat=4):
        if not np.isclose(marginal[p, q, r, s], -1*marginal[q, p, r, s]):
            return False
        if not np.isclose(marginal[p, q, r, s], -1*marginal[q, p, r, s]):
            return False
        if not np.isclose(marginal[p, q, r, s], -1*marginal[p, q, s, r]):
            return False
        if not np.isclose(marginal[p, q, r, s], marginal[q, p, s, r]):
            return False
    return True


