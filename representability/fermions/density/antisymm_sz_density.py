"""
Utilities for returning marginals with SZ spin-adapting.
The alpha-alpha basis functions are symmetric basis functions.
This is the spin adapting that appears in most of the 2-RDM papers.

More details can be found in PhysRevA.72.052505

The 1-RDMs are block diagonal matrices corresponding to their spin-index
(alpha or beta).  The 2-RDM and 2-Hole-RDM contain three blocks
aa, bb, ab blocks with (r*(r - 1)/2), r*(r - 1)/2, and r**2) linear size
where r is the spatial basis function rank.
"""
import numpy as np
from itertools import product
from grove.alpha.fermion_transforms.jwtransform import JWTransform
from representability.fermions.density.symm_sz_density import SymmOrbitalDensity
from representability.fermions.density.antisymm_sz_maps import map_d2anti_d1_sz, map_d2symm_d1_sz


class AntiSymmOrbitalDensity(SymmOrbitalDensity):

    def __init__(self, rho, dim, transform=JWTransform()):
        """
        Full Sz symmetry adapting

        The internal `_tensor_construct()` method is inherited from the
        `SymmOrbitalDensity` object.  The methods `construct_opdm()` and
        `construc_ohdm()` are also inherited as their behavior is the same.

        The full symmetry adapting requires a different structure and data
        abstraction for the marginals.

        :param rho: N-qubit density matrix
        :param dim: single particle basis rank (spin-orbital rank)
        :param transform: Fermionc to qubit transform object
        """
        super(AntiSymmOrbitalDensity, self).__init__(rho, dim)
        self.transform = transform
        self.dim = dim

    def construct_tpdm(self):
        """
        Return the two-particle density matrix
<psi|a_{p, sigma}^{\dagger}a_{q, sigma'}^{\dagger}a_{s, sigma'}a_{r, sigma}|psi> """
        conjugate = [-1, -1, 1, 1]
        spin_blocks = [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]]
        rdms = list(map(lambda x: self._tensor_construct(len(conjugate), conjugate,
                                                    x), spin_blocks))
        # antisymmetric basis dimension
        ab_dim = int((self.dim/2) * (self.dim/2 - 1) / 2)
        d2_aa = np.zeros((ab_dim, ab_dim))
        d2_bb = np.zeros_like(d2_aa)
        d2_ab = np.zeros((int((self.dim / 2)**2), int((self.dim / 2)**2)))

        # build basis look up table
        bas_aa = {}
        bas_ab = {}
        cnt_aa = 0
        cnt_ab = 0
        # iterate over spatial orbital indices
        for p, q in product(range(int(self.dim/2)), repeat=2):
            if q > p:
                bas_aa[(p, q)] = cnt_aa
                cnt_aa += 1
            bas_ab[(p, q)] = cnt_ab
            cnt_ab += 1

        # iterate over 4 spatial orbital indices
        for i, j, k, l in product(range(int(self.dim/2)), repeat=4):
            d2_ab[bas_ab[(i, j)], bas_ab[(k, l)]] = rdms[2][i, j, k, l].real

            if i < j and k < l:
                d2_aa[bas_aa[(i, j)], bas_aa[(k, l)]] = rdms[0][i, j, k, l].real - \
                                                        rdms[0][i, j, l, k].real - \
                                                        rdms[0][j, i, k, l].real + \
                                                        rdms[0][j, i, l, k].real

                d2_bb[bas_aa[(i, j)], bas_aa[(k, l)]] = rdms[1][i, j, k, l].real - \
                                                        rdms[1][i, j, l, k].real - \
                                                        rdms[1][j, i, k, l].real + \
                                                        rdms[1][j, i, l, k].real

                d2_aa[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5
                d2_bb[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5

        return d2_aa, d2_bb, d2_ab, [bas_aa, bas_ab]

    def construct_thdm(self):
        """
        Return the two-hole density matrix

        <psi|a_{p, sigma}a_{q, sigma'}a_{s, sigma'}^{\dagger}a_{r, sigma}^{\dagger}|psi>
        """
        conjugate = [1, 1, -1, -1]
        spin_blocks = [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]]
        rdms = list(map(lambda x: self._tensor_construct(len(conjugate), conjugate,
                                                    x), spin_blocks))

        # antisymmetric basis dimension
        ab_dim = int((self.dim/2) * (self.dim/2 - 1) / 2)
        sp_dim = int(self.dim/2)**2
        div2dim = int(self.dim / 2)
        q2_aa = np.zeros((ab_dim, ab_dim))
        q2_bb = np.zeros_like(q2_aa)
        q2_ab = np.zeros((sp_dim, sp_dim))

        # build basis look up table
        bas_aa = {}
        bas_ab = {}
        cnt_aa = 0
        cnt_ab = 0
        # iterate over spatial orbital indices
        for p, q in product(range(div2dim), repeat=2):
            if q > p:
                bas_aa[(p, q)] = cnt_aa
                cnt_aa += 1
            bas_ab[(p, q)] = cnt_ab
            cnt_ab += 1

        # iterate over 4 spatial orbital indices
        for i, j, k, l in product(range(div2dim), repeat=4):
            q2_ab[bas_ab[(i, j)], bas_ab[(k, l)]] = rdms[2][i, j, k, l].real

            if i < j and k < l:
                q2_aa[bas_aa[(i, j)], bas_aa[(k, l)]] = rdms[0][i, j, k, l].real - \
                                                        rdms[0][i, j, l, k].real - \
                                                        rdms[0][j, i, k, l].real + \
                                                        rdms[0][j, i, l, k].real

                q2_bb[bas_aa[(i, j)], bas_aa[(k, l)]] = rdms[1][i, j, k, l].real - \
                                                        rdms[1][i, j, l, k].real - \
                                                        rdms[1][j, i, k, l].real + \
                                                        rdms[1][j, i, l, k].real


                q2_aa[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5
                q2_bb[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5

        return q2_aa, q2_bb, q2_ab, [bas_aa, bas_ab]

    def construct_tpdm_error_matrix(self, error_tpdm):
        """
        Construct the error matrix for a block of the marginal

        [I] [E]
        [E*] [0]

        where I is the identity matrix, E is ^{2}D_{meas} - ^{2}D_{var}, F is a matrix of Free variables

        :param corrupted_tpdm:
        :return:
        """
        if np.ndim(error_tpdm) != 2:
            raise TypeError("corrupted_tpdm needs to be a matrix")

        dim = int(error_tpdm.shape[0])
        top_row_emat = np.hstack((np.eye(dim), error_tpdm))
        bottom_row_emat = np.hstack((error_tpdm.T, np.zeros((dim, dim))))
        error_schmidt_matrix = np.vstack((top_row_emat, bottom_row_emat))

        return error_schmidt_matrix


def check_d2_d1_sz_antisymm(tpdm, opdm, normalization, bas):
    """
    check the contractive map from d2 to d1
    """
    opdm_test = map_d2anti_d1_sz(tpdm, normalization, bas, opdm.shape[0])
    return np.allclose(opdm_test, opdm)


def check_d2_d1_sz_symm(tpdm, opdm, normalization, bas):
    """
    check the contraction from d2 to d1 for aa and bb matrices
    """
    return np.allclose(opdm, map_d2symm_d1_sz(tpdm, normalization, bas, opdm.shape[0]))


def unspin_adapt(d2aa, d2bb, d2ab):
    """
    Transform a sz_spin-adapted set of 2-RDMs back to the spin-orbtal 2-RDM

    :param d2aa: alpha-alpha block of the 2-RDM.  Antisymmetric basis functions
                 are assumed for this block. block size is r_{s} * (r_{s} - 1)/2
                 where r_{s} is the number of spatial basis functions
    :param d2bb: beta-beta block of the 2-RDM.  Antisymmetric basis functions
                 are assumed for this block. block size is r_{s} * (r_{s} - 1)/2
                 where r_{s} is the number of spatial basis functions
    :param d2ab: alpha-beta block of the 2-RDM. no symmetry adapting is perfomred
                 on this block.  Map directly back to spin-orbital components.
                 This block should have linear dimension r_{s}^{2} where r_{S}
                 is the number of spatial basis functions.
    :return: four-tensor representing the spin-orbital density matrix.
    """
    sp_dim = int(np.sqrt(d2ab.shape[0]))
    so_dim = 2 * sp_dim
    tpdm = np.zeros((so_dim, so_dim, so_dim, so_dim), dtype=complex)

    # build basis look up table
    bas_aa = {}
    bas_ab = {}
    cnt_aa = 0
    cnt_ab = 0
    for p, q in product(range(sp_dim), repeat=2):
        if q > p:
            bas_aa[(p, q)] = cnt_aa
            cnt_aa += 1
        bas_ab[(p, q)] = cnt_ab
        cnt_ab += 1

    # map the d2aa and d2bb back to the spin-orbital 2-RDM
    for p, q, r, s in product(range(sp_dim), repeat=4):
        if p < q and r < s:
            tpdm[2 * p, 2 * q, 2 * r, 2 * s] =  0.5 * d2aa[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * p, 2 * q, 2 * s, 2 * r] = -0.5 * d2aa[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q, 2 * p, 2 * r, 2 * s] = -0.5 * d2aa[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q, 2 * p, 2 * s, 2 * r] =  0.5 * d2aa[bas_aa[(p, q)], bas_aa[(r, s)]]

            tpdm[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] =  0.5 * d2bb[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * p + 1, 2 * q + 1, 2 * s + 1, 2 * r + 1] = -0.5 * d2bb[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q + 1, 2 * p + 1, 2 * r + 1, 2 * s + 1] = -0.5 * d2bb[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q + 1, 2 * p + 1, 2 * s + 1, 2 * r + 1] =  0.5 * d2bb[bas_aa[(p, q)], bas_aa[(r, s)]]

        tpdm[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = d2ab[bas_ab[(p, q)], bas_ab[(r, s)]]
        tpdm[2 * q + 1, 2 * p, 2 * r, 2 * s + 1] = -1 * d2ab[bas_ab[(p, q)], bas_ab[(r, s)]]
        tpdm[2 * p, 2 * q + 1, 2 * s + 1, 2 * r] = -1 * d2ab[bas_ab[(p, q)], bas_ab[(r, s)]]
        tpdm[2 * q + 1, 2 * p, 2 * s + 1, 2 * r] = d2ab[bas_ab[(p, q)], bas_ab[(r, s)]]

    return tpdm







