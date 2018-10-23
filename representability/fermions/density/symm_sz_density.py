"""
Utilities for returning marginals with SZ spin-adapting
all basis functions are symmetric meaning we will have
elements that are necessarily zero in the marginals.
"""
import numpy as np
from itertools import product
from referenceqvm.unitary_generator import tensor_up
from grove.alpha.fermion_transforms.jwtransform import JWTransform
from representability.fermions.density.density import Density
from representability.fermions.density.symm_sz_maps import map_d1_q1, map_d2_d1, map_d2_q2, map_d2_q2_ab, \
                                                           map_d2_g2, map_d2_g2_sz


class SymmOrbitalDensity(Density):

    def __init__(self, rho, dim, transform=JWTransform()):
        """
        :param rho: N-qubit density matrix
        :param dim: single particle basis rank (spin-orbital rank)
        :param transform: Fermionc to qubit transform object
        """
        super(SymmOrbitalDensity, self).__init__(rho)
        self.transform = transform
        self.dim = dim

    def _tensor_construct(self, rank, conjugates, updown):
        """
        General procedure for evaluating the expected value of second quantized ops

        :param Int rank: number of second quantized operators to product out
        :param List conjugates: Indicator of the conjugate type of the second
                                quantized operator
        :param List updown: SymmOrbitalDensity matrices are index by spatial orbital
                            Indices.  This value is self.dim/2 for Fermionic systems.
                            When projecting out expected values to form the marginals,
                            we need to know if the spin-part of the spatial basis
                            funciton.  updown is a list corresponding with the
                            conjugates telling us if we are projecting an up spin or
                            down spin.  0 is for up. 1 is for down.

                            Example: to get the 1-RDM alpha-block we would pass the
                            following:

                                rank = 2, conjugates = [-1, 1], updown=[0, 0]

        :returns: a tensor of (rank) specified by input
        :rytpe: np.ndarray
        """
        # self.dim/2 because rank is now spatial basis function rank
        tensor = np.zeros(tuple([int(self.dim/2)] * rank), dtype=complex)
        for tindices in product(range(int(self.dim/2)), repeat=rank):
            # get the spatial indices
            spin_free_indices = list(map(lambda x: 2 * tindices[x] + updown[x],
                                    range(len(tindices))))

            pauli_proj_op = self.transform.product_ops(
                            list(spin_free_indices[:int(rank/2)] +
                                 spin_free_indices[int(rank/2):][::-1]), conjugates)

            lifted_op = tensor_up(pauli_proj_op, self.num_qubits)
            element = np.trace(lifted_op.dot(self.rho))
            tensor[tindices] = element
        return tensor


    def construct_opdm(self):
        """
        Return the one-particle density matrix

        <psi|a_{p, sigma}^{\dagger}a_{q, sigma}|psi>
        """
        conjugate = [-1, 1]
        spin_blocks = [[0, 0], [1, 1]]
        rdms = map(lambda x: self._tensor_construct(len(conjugate), conjugate,
                                                    x), spin_blocks)
        return rdms

    def construct_ohdm(self):
        """
        Return the one-hole density matrix

        <psi|a_{p, sigma}a_{q, sigma}^{\dagger}|psi>
        """
        conjugate = [1, -1]
        spin_blocks = [[0, 0], [1, 1]]
        rdms = map(lambda x: self._tensor_construct(len(conjugate), conjugate,
                                                    x), spin_blocks)

        return rdms


    def construct_tpdm(self):
        """
        Return the two-particle density matrix

        <psi|a_{p, sigma}^{\dagger}a_{q, sigma'}^{\dagger}a_{s, sigma'}a_{r, sigma}|psi>
        """
        conjugate = [-1, -1, 1, 1]
        spin_blocks = [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]]
        rdms = map(lambda x: self._tensor_construct(len(conjugate), conjugate,
                                                    x), spin_blocks)
        return rdms


    def construct_thdm(self):
        """
        Return the two-hole density matrix

        <psi|a_{p, sigma}a_{q, sigma'}a_{s, sigma'}^{\dagger}a_{r, sigma}^{\dagger}|psi>
        """
        conjugate = [1, 1, -1, -1]
        spin_blocks = [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]]
        rdms = map(lambda x: self._tensor_construct(len(conjugate), conjugate,
                                                    x), spin_blocks)

        return rdms


    def construct_phdm(self):
        """
        Return the particle-hole density matrix

        <psi|a_{p, sigma}^{\dagger}a_{q, sigma'}a_{s, sigma'}^{\dagger}a_{r, sigma}|psi>
        """
        conjugate = [-1, 1, -1, 1]
        spin_blocks = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1],
                       [0, 0, 1, 1], [1, 1, 0, 0]]
        rdms = list(map(lambda x: self._tensor_construct(len(conjugate), conjugate,
                                                    x), spin_blocks))

        # unfortunately, this will not do in terms of a tensor structure.  Yes, the code works but when it is unrolled
        # into a matrix on the SDP side there is no guarantee of the correct ordering.

        # g2_aabb = np.zeros((2, 2, self.dim/2, self.dim/2,
        #                     self.dim/2, self.dim/2), dtype=complex)
        # g2_aabb[0, 0, :, :, :, :] = rdms[2]
        # g2_aabb[1, 1, :, :, :, :] = rdms[3]
        # g2_aabb[0, 1, :, :, :, :] = rdms[4]
        # g2_aabb[1, 0, :, :, :, :] = rdms[5]

        dim = int(self.dim / 2)
        mm = dim ** 2
        g2_aabb = np.zeros((2*mm, 2*mm))
        # unroll into the blocks
        for p, q, r, s in product(range(int(self.dim/2)), repeat=4):
            g2_aabb[p * dim + q, r * dim + s] = rdms[2][p, q, r, s].real
            g2_aabb[p * dim + q + dim**2, r * dim + s + dim**2] = rdms[3][p, q, r, s].real
            g2_aabb[p * dim + q, r * dim + s + dim**2] = rdms[4][p, q, r, s].real
            g2_aabb[p * dim + q + dim**2, r * dim + s] = rdms[5][p, q, r, s].real

        return [rdms[0], rdms[1], g2_aabb]


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


def check_d2_d1_map(tpdm, opdm, normalization):
    """
    Given a 2-RDM and a 1-RDM check the contraction of the 2-RDM to the 1-RDM

    :param tpdm: rank-4 tensor that is the 2-RDM
    :param opdm: rank-2 tensor that is the 1-RDM
    :param normalization: normalziation constant to use
    :returns: True/False if mapping is good
    :rtype: Bool
    """
    opdm_test = map_d2_d1(tpdm, normalization)
    return np.allclose(opdm_test, opdm)


def check_d2_q2_map(tpdm, tqdm, opdm):
    """
    """
    tqdm_test = map_d2_q2(tpdm, opdm)
    return np.allclose(tqdm_test, tqdm)

def check_d2_q2_ab_map(tpdm, tqdm, opdm_a, opdm_b):
    """
    """
    tqdm_test = map_d2_q2_ab(tpdm, opdm_a, opdm_b)
    return np.allclose(tqdm_test, tqdm)

def check_d2_g2_map(tpdm, phdm, opdm):
    """
    """
    phdm_test = map_d2_g2(tpdm, opdm)
    return np.allclose(phdm_test, phdm)

def check_d2_g2_sz_map(tpdm, tgdm_aabb, opdm, block='aaaa'):
    """
    """
    return np.allclose(tgdm_aabb, map_d2_g2_sz(tpdm, opdm, block))
