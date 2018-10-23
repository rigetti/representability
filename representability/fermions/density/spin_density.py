"""
Utilities for returning marginals in the spin-orbital
basis
"""
import numpy as np
from itertools import product
from referenceqvm.unitary_generator import tensor_up
from grove.alpha.fermion_transforms.jwtransform import JWTransform
from representability.fermions.density.density import Density
from representability.fermions.utils import four_tensor2matrix
from representability.fermions.density.spin_maps import (map_d1_q1, map_d2_d1,
                                                         map_d2_q2, map_d2_g2)


class SpinOrbitalDensity(Density):

    def __init__(self, rho, dim, transform=JWTransform()):
        """
        :param rho: N-qubit density matrix
        :param dim: single particle basis rank
        :param transform: Fermionc to qubit transform object
        """
        super(SpinOrbitalDensity, self).__init__(rho)
        self.transform = transform
        self.dim = dim

    def _tensor_construct(self, rank, conjugates):
        """
        General procedure for evaluating the expected value of second quantized ops

        General procedure for finding p-rank correlators such as the 1-RDM or 2-RDM
        :param Int rank: number of second quantized operators to product out
        :param List conjugates: Indicator of the conjugate type of the second quantized operator
        :returns: a tensor of (rank) specified by input
        :rytpe: np.ndarray
        """
        tensor = np.zeros(tuple([self.dim] * rank), dtype=complex)
        for tindices in product(range(self.dim), repeat=rank):
            print(tindices)
            pauli_proj_op = self.transform.product_ops(list(tindices[:int(rank/2)] + tindices[int(rank/2):][::-1]), conjugates)
            lifted_op = tensor_up(pauli_proj_op, self.num_qubits)
            # element = np.trace(lifted_op.dot(self.rho))
            element = (lifted_op.dot(self.rho)).diagonal().sum()
            tensor[tindices] = element
        return tensor

    def construct_opdm(self):
        """
        Return the one-particle density matrix

        <psi|a_{p}^{\dagger}a_{q}|psi>
        """
        conjugate = [-1, 1]
        return self._tensor_construct(len(conjugate), conjugate)

    def construct_ohdm(self):
        """
        Return the one-hole density matrix

        <psi|a_{p}a_{q}^{\dagger}|psi>
        """
        conjugate = [1, -1]
        return self._tensor_construct(len(conjugate), conjugate)

    def construct_tpdm(self):
        """
        Return the two-particle density matrix

        <psi|a_{p}^{\dagger}a_{q}^{\dagger}a_{s}a_{r}|psi>
        """
        conjugate = [-1, -1, 1, 1]
        return self._tensor_construct(len(conjugate), conjugate)

    def construct_thdm(self):
        """
        Return the two-hole density matrix

        <psi|a_{p}a_{q}a_{s}^{\dagger}a_{r}^{\dagger}|psi>
        """
        conjugate = [1, 1, -1, -1]
        return self._tensor_construct(len(conjugate), conjugate)

    def construct_phdm(self):
        """
        Return the particle-hole density matrix

        <psi|a_{p}^{\dagger}a_{q}a_{s}^{\dagger}a_{r}|psi>
        """
        conjugate = [-1, 1, -1, 1]
        return self._tensor_construct(len(conjugate), conjugate)

    def construct_tpdm_error_matrix(self, error_tpdm):
        """
        Construct the error tensor

        Structure of tensor is a large matrix with the following block structure

        [I] [E]
        [E] [F]

        where I is the identity matrix, E is ^{2}D_{meas} - ^{2}D_{var}, F is a matrix of Free variables

        :return:
        """

        # if our tensor is rank 4 then we need to reshape
        if np.ndim(error_tpdm) == 4:
            error_tpdm_matrix = four_tensor2matrix(error_tpdm)
        else:
            error_tpdm_matrix = np.copy(error_tpdm)

        dim = error_tpdm_matrix.shape[0]

        top_row_emat = np.hstack((np.eye(dim), error_tpdm_matrix))
        bottom_row_emat = np.hstack((error_tpdm_matrix.T, np.zeros((dim, dim), dtype=error_tpdm.dtype)))
        error_schmidt_matrix = np.vstack((top_row_emat, bottom_row_emat))

        return error_schmidt_matrix







def check_d1_q1_map(opdm, ohdm):
    """
    Given a 1-RDM and 1-H-RDM check their mapping

    :param opdm: rank-2 tensor that is the 1-RDM
    :param ohdm: rank-2 tensor that is the 1-H-RDM
    :returns: Boolean of appropriate mapping
    :rtype: Bool
    """
    ohdm_test = map_d1_q1(opdm)
    return np.allclose(ohdm_test, ohdm)


def check_d2_d1_map(tpdm, opdm):
    """
    Given a 2-RDM and a 1-RDM check the contraction of the 2-RDM to the 1-RDM

    :param tpdm: rank-4 tensor that is the 2-RDM
    :param opdm: rank-2 tensor that is the 1-RDM
    :returns: True/False if mapping is good
    :rtype: Bool
    """
    opdm_test = map_d2_d1(tpdm, np.trace(opdm))
    return np.allclose(opdm_test, opdm)


def check_d2_q2_map(tpdm, tqdm, opdm):
    """
    """
    tqdm_test = map_d2_q2(tpdm, opdm)
    return np.allclose(tqdm_test, tqdm)


def check_d2_g2_map(tpdm, phdm, opdm):
    """
    """
    phdm_test = map_d2_g2(tpdm, opdm)
    return np.allclose(phdm_test, phdm)


