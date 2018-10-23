"""
Constraint generator and surrounding utilities for marginal problem in block diagonal
form by SZ operator.  antisymm_sz name comes from the fact that the alpha-alpha
and beta-beta blocks use antisymmetric basis functions as is common in the
2-RDM literature.  This is the most efficient non-redundant form for the SZ
symmetry adpating available.
"""
import sys
import numpy as np
from itertools import product
from representability.dualbasis import DualBasisElement, DualBasis


def _trace_map(tname, dim, normalization):
    dbe = DualBasisElement()
    for i, j in product(range(dim), repeat=2):
        if i < j:
            dbe.add_element(tname, (i, j, i, j), 1.0)
    dbe.dual_scalar = normalization
    return dbe


def trace_d2_aa(dim, Na):
    return _trace_map('cckk_aa', dim, Na * (Na - 1))


def trace_d2_bb(dim, Nb):
    return _trace_map('cckk_bb', dim, Nb * (Nb - 1))


def trace_d2_ab(dim, Na, Nb):
    dbe = DualBasisElement()
    for i, j in product(range(dim), repeat=2):
        dbe.add_element('cckk_ab', (i, j, i, j), 1.0)
    dbe.dual_scalar = Na * Nb
    return dbe


def s_representability_d2ab(dim, N, M, S):
    """
    Constraint for S-representability

    PHYSICAL REVIEW A 72, 052505 2005


    :param dim: number of spatial basis functions
    :param N: Total number of electrons
    :param M: Sz expected value
    :param S: S(S + 1) is eigenvalue of S^{2}
    :return:
    """
    dbe = DualBasisElement()
    for i, j in product(range(dim), repeat=2):
        dbe.add_element('cckk_ab', (i, j, j, i), 1.0)
    dbe.dual_scalar = N/2.0 + M**2 - S*(S + 1)
    return dbe


def sz_representability(dim, M):
    """
    Constraint for S_z-representability

    Helgaker, Jorgensen, Olsen. Sz is one-body RDM constraint

    :param dim: number of spatial basis functions
    :param M: Sz expected value
    :return:
    """
    dbe = DualBasisElement()
    for i in range(dim):
        dbe.add_element('ck_a', (i, i), 0.5)
        dbe.add_element('ck_b', (i, i), -0.5)
    dbe.dual_scalar = M
    return dbe


def d2ab_d1a_mapping(dim, Nb):
    """
    Map the d2_spin-adapted 2-RDM to the D1 rdm

    :param Nb: number of beta electrons
    :param dim:
    :return:
    """
    return _contraction_base('cckk_ab', 'ck_a', dim, Nb, 0)


def d2ab_d1b_mapping(dim, Na):
    """
    Map the d2_spin-adapted 2-RDM to the D1 rdm

    :param Nb: number of beta electrons
    :param dim:
    :return:
    """
    db = DualBasis()
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            for r in range(dim):
                dbe.add_element('cckk_ab', (r, i, r, j), 0.5)
                dbe.add_element('cckk_ab', (r, j, r, i), 0.5)

            dbe.add_element('ck_b', (i, j), -0.5 * Na)
            dbe.add_element('ck_b', (j, i), -0.5 * Na)
            dbe.dual_scalar = 0

            dbe.simplify()
            db += dbe

    return db


def d2aa_d1a_mapping(dim, Na):
    """
    Map the d2_spin-adapted 2-RDM to the D1 rdm

    :param Nb: number of beta electrons
    :param dim:
    :return:
    """
    db = DualBasis()
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            for r in range(dim):
                # Not in the basis because always zero
                if i == r or j == r:
                    continue
                else:
                    sir = 1 if i < r else -1
                    sjr = 1 if j < r else -1
                    ir_pair = (i, r) if i < r else (r, i)
                    jr_pair = (j, r) if j < r else (r, j)
                    if i == j:
                        dbe.add_element('cckk_aa', (ir_pair[0], ir_pair[1], jr_pair[0], jr_pair[1]), sir * sjr * 0.5)
                    else:
                        # TODO: Remember why I derived a factor of 0.25 (0.5 above) for this equation.
                        dbe.add_element('cckk_aa', (ir_pair[0], ir_pair[1], jr_pair[0], jr_pair[1]), sir * sjr * 0.25)
                        dbe.add_element('cckk_aa', (jr_pair[0], jr_pair[1], ir_pair[0], ir_pair[1]), sir * sjr * 0.25)

            dbe.add_element('ck_a', (i, j), -0.5 * (Na - 1))
            dbe.add_element('ck_a', (j, i), -0.5 * (Na - 1))
            dbe.dual_scalar = 0

            dbe.simplify()
            db += dbe

    return db


def d2bb_d1b_mapping(dim, Nb):
    """
    Map the d2_spin-adapted 2-RDM to the D1 rdm

    :param Nb: number of beta electrons
    :param dim:
    :return:
    """
    db = DualBasis()
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            for r in range(dim):
                # Not in the basis because always zero
                if i == r or j == r:
                    continue
                else:
                    sir = 1 if i < r else -1
                    sjr = 1 if j < r else -1
                    ir_pair = (i, r) if i < r else (r, i)
                    jr_pair = (j, r) if j < r else (r, j)
                    if i == j:
                        dbe.add_element('cckk_bb', (ir_pair[0], ir_pair[1], jr_pair[0], jr_pair[1]), sir * sjr * 0.5)
                    else:
                        # TODO: Remember why I derived a factor of 0.25 (0.5 above) for this equation.
                        dbe.add_element('cckk_bb', (ir_pair[0], ir_pair[1], jr_pair[0], jr_pair[1]), sir * sjr * 0.25)
                        dbe.add_element('cckk_bb', (jr_pair[0], jr_pair[1], ir_pair[0], ir_pair[1]), sir * sjr * 0.25)

            dbe.add_element('ck_b', (i, j), -0.5 * (Nb - 1))
            dbe.add_element('ck_b', (j, i), -0.5 * (Nb - 1))
            dbe.dual_scalar = 0

            dbe.simplify()
            db += dbe

    return db
# TODO: Implement trace constraints on these spin-blocks (PHYSICAL REVIEW A 72, 052505 2005,...
# E. Perez-Romero, L. M. Tel, and C. Valdemoro, Int. J. Quantum Chem. 61, 55 19970)
# def d2aa_d1a_mapping(Na, dim):
#     """
#     Map the d2_spin-adapted 2-RDM to the D1 rdm
#
#     :param Nb: number of beta electrons
#     :param dim:
#     :return:
#     """
#     return _contraction_base('cckk_aa', 'ck_a', dim, Na - 1, 1)
#
#
# def d2bb_d1b_mapping(Nb, dim):
#     """
#     Map the d2_spin-adapted 2-RDM to the D1 rdm
#
#     :param Nb: number of beta electrons
#     :param dim:
#     :return:
#     """
#     return _contraction_base('cckk_bb', 'ck_b', dim, Nb - 1, 1)


def _contraction_base(tname_d2, tname_d1, dim, normalization, offset):
    db = DualBasis()
    for i in range(dim):
        for j in range(i + offset, dim):
            dbe = DualBasisElement()
            for r in range(dim):
                dbe.add_element(tname_d2, (i, r, j, r), 0.5)
                dbe.add_element(tname_d2, (j, r, i, r), 0.5)

            dbe.add_element(tname_d1, (i, j), -0.5 * normalization)
            dbe.add_element(tname_d1, (j, i), -0.5 * normalization)
            dbe.dual_scalar = 0

            dbe.simplify()
            db += dbe

    return db


def _d1_q1_mapping(tname_d1, tname_q1, dim):
    db = DualBasis()
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            if i != j:
                dbe.add_element(tname_d1, (i, j), 0.5)
                dbe.add_element(tname_d1, (j, i), 0.5)
                dbe.add_element(tname_q1, (i, j), 0.5)
                dbe.add_element(tname_q1, (j, i), 0.5)
                dbe.dual_scalar = 0.0
            else:
                dbe.add_element(tname_d1, (i, j), 1.0)
                dbe.add_element(tname_q1, (i, j), 1.0)
                dbe.dual_scalar = 1.0

            db += dbe.simplify()

    return db


def d1a_d1b_mapping(tname_d1a, tname_d1b, dim):
    db = DualBasis()
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            if i != j:
                dbe.add_element(tname_d1a, (i, j), 0.5)
                dbe.add_element(tname_d1a, (j, i), 0.5)
                dbe.add_element(tname_d1b, (i, j), -0.5)
                dbe.add_element(tname_d1b, (j, i), -0.5)
                dbe.dual_scalar = 0.0
            else:
                dbe.add_element(tname_d1a, (i, j), 1.0)
                dbe.add_element(tname_d1b, (i, j), -1.0)
                dbe.dual_scalar = 0.0

            db += dbe.simplify()

    return db


def d1a_q1a_mapping(dim):
    """
    Generate the dual basis elements for spin-blocks of d1 and q1

    :param dim: spatial basis rank
    :return:
    """
    return _d1_q1_mapping('ck_a', 'kc_a', dim)


def d1b_q1b_mapping(dim):
    """
    Generate the dual basis elements for spin-blocks of d1 and q1

    :param dim: spatial basis rank
    :return:
    """
    return _d1_q1_mapping('ck_b', 'kc_b', dim)


# TODO: Modularize the spin-block constraints for Q2
def d2_q2_mapping(dim):
    """
    Map each d2 block to the q2 block

    :param dim: rank of spatial single-particle basis
    :return:
    """
    krond = np.eye(dim)
    def d2q2element(p, q, r, s, factor, tname_d1_1, tname_d1_2, tname_d2, tname_q2):
        dbe = DualBasisElement()
        dbe.add_element(tname_d1_1, (p, r), 2.0 * krond[q, s] * factor)
        dbe.add_element(tname_d1_2, (q, s), 2.0 * krond[p, r] * factor)
        dbe.add_element(tname_d1_1, (p, s), -2.0 * krond[r, q] * factor)
        dbe.add_element(tname_d1_2, (q, r), -2.0 * krond[p, s] * factor)
        dbe.add_element(tname_q2, (r, s, p, q), 1.0 * factor)
        dbe.add_element(tname_d2, (p, q, r, s), -1.0 * factor)

        # remember the negative sign because AX = b
        dbe.dual_scalar = -2.0 * krond[s, p]*krond[r, q] * factor + 2.0 * krond[q, s] * krond[r, p] * factor
        return dbe

    def d2q2element_ab(p, q, r, s, factor, tname_d1_1, tname_d1_2, tname_d2, tname_q2):
        if tname_d1_1 != 'ck_a':
            raise TypeError("For some reason I am expecting a ck_a. Ask Nick")

        dbe = DualBasisElement()
        dbe.add_element(tname_d1_1, (p, r), krond[q, s] * factor)
        dbe.add_element(tname_d1_2, (q, s), krond[p, r] * factor)
        dbe.add_element(tname_q2, (r, s, p, q), 1.0 * factor)
        dbe.add_element(tname_d2, (p, q, r, s), -1.0 * factor)
        dbe.dual_scalar = krond[q, s]*krond[p, r] * factor
        return dbe

    db = DualBasis()
    d2_names = ['cckk_aa', 'cckk_bb', 'cckk_ab']
    q2_names = ['kkcc_aa', 'kkcc_bb', 'kkcc_ab']
    d1_names_1 = ['ck_a', 'ck_b', 'ck_a']
    d1_names_2 = ['ck_a', 'ck_b', 'ck_b']
    dual_basis_list = []
    for key in zip(d1_names_1, d1_names_2, d2_names, q2_names):
        d1_1, d1_2, d2_n, q2_n = key
        for p, q, r, s in product(range(dim), repeat=4):
            if (d2_n == 'cckk_aa' or d2_n == 'cckk_bb') and p < q and r < s and p * dim + q <= r * dim + s:
                dbe_1 = d2q2element(p, q, r, s, 0.5, d1_1, d1_2, d2_n, q2_n)
                dbe_2 = d2q2element(r, s, p, q, 0.5, d1_1, d1_2, d2_n, q2_n)
                # db += dbe_1.join_elements(dbe_2)
                dual_basis_list.append(dbe_1.join_elements(dbe_2))
            elif d2_n == 'cckk_ab' and p * dim + q <= r * dim + s:
                dbe_1 = d2q2element_ab(p, q, r, s, 0.5, d1_1, d1_2, d2_n, q2_n)
                dbe_2 = d2q2element_ab(r, s, p, q, 0.5, d1_1, d1_2, d2_n, q2_n)
                # db += dbe_1.join_elements(dbe_2)
                dual_basis_list.append(dbe_1.join_elements(dbe_2))

    return DualBasis(elements=dual_basis_list)


# TODO: Modularize the spin-block constraints for G2
def d2_g2_mapping(dim):
    """
    Map each d2 blcok to the g2 blocks

    :param dim: rank of spatial single-particle basis
    :return:
    """
    krond = np.eye(dim)
    # d2 -> g2
    def g2d2map_aabb(p, q, r, s, dim, key, factor=1.0):
        """
        Accept pqrs of G2 and map to D2
        """
        dbe = DualBasisElement()
        # this is ugly.  :(
        quad = {'aabb': [0, 1], 'bbaa': [1, 0]}
        dbe.add_element('ckck_aabb', (p * dim + q + quad[key][0]*dim**2, r * dim + s + quad[key][1]*dim**2),
                        1.0 * factor)
        dbe.add_element('ckck_aabb', (r * dim + s + quad[key[::-1]][0]*dim**2, p * dim + q + quad[key[::-1]][1]*dim**2),
                        1.0 * factor)

        dbe.add_element('cckk_ab', (p, s, q, r), -1.0 * factor)
        dbe.add_element('cckk_ab', (q, r, p, s), -1.0 * factor)
        dbe.dual_scalar = 0.0
        return dbe

    def g2d2map_ab(p, q, r, s, key, factor=1.0):
        dbe = DualBasisElement()
        if key == 'ab':
            dbe.add_element('ck_' + key[0], (p, r), krond[q, s] * factor)
            dbe.add_element('cckk_' + key, (p, s, r, q), -1.0 * factor)
        elif key == 'ba':
            dbe.add_element('ck_' + key[0], (p, r), krond[q, s] * factor)
            dbe.add_element('cckk_ab', (s, p, q, r), -1.0 * factor)
        else:
            raise TypeError("I only accept ab or ba blocks")

        dbe.add_element('ckck_' + key, (p, q, r, s), -1.0 * factor)
        dbe.dual_scalar = 0.0
        return dbe

    def g2d2map_aa_or_bb(p, q, r, s, dim, key, factor=1.0):
        """
        Accept pqrs of G2 and map to D2
        """
        dbe = DualBasisElement()
        quad = {'aa': [0, 0], 'bb': [1, 1]}
        dbe.add_element('ckck_aabb', (p * dim + q + quad[key][0]*dim**2, r * dim + s + quad[key][1]*dim**2),
                        -1.0 * factor)
        dbe.add_element('ck_' + key[0], (p, r), krond[q, s] * factor)
        if p != s and r != q:
            gem1 = tuple(sorted([p, s]))
            gem2 = tuple(sorted([r, q]))
            parity = (-1)**(p < s) * (-1)**(r < q)
            dbe.add_element('cckk_' + key, (gem1[0], gem1[1], gem2[0], gem2[1]), parity * -0.5 * factor)

        dbe.dual_scalar = 0
        return dbe

    db = DualBasis()
    # do aa_aa block then bb_block
    dual_basis_list = []
    for key in ['bb', 'aa']:
        for p, q, r, s in product(range(dim), repeat=4):
            if p * dim + q <= r * dim + s:
                dbe_1 = g2d2map_aa_or_bb(p, q, r, s, dim, key, factor=0.5)
                dbe_2 = g2d2map_aa_or_bb(r, s, p, q, dim, key, factor=0.5)
                # db += dbe_1.join_elements(dbe_2)
                dual_basis_list.append(dbe_1.join_elements(dbe_2))

    # this constraint is over the entire block!
    for key in ['aabb']:
        for p, q, r, s in product(range(dim), repeat=4):
            dbe = g2d2map_aabb(p, q, r, s, dim, key, factor=1.0)
            # db += dbe
            dual_basis_list.append(dbe)

    # # ab ba blocks of G2
    for key in ['ab', 'ba']:
        for p, q, r, s in product(range(dim), repeat=4):
            if p * dim + q <= r * dim + s:
                dbe_1 = g2d2map_ab(p, q, r, s, key, factor=0.5)
                dbe_2 = g2d2map_ab(r, s, p, q, key, factor=0.5)
                # db += dbe_1.join_elements(dbe_2)
                dual_basis_list.append(dbe_1.join_elements(dbe_2))

    return DualBasis(elements=dual_basis_list)


def d2_e2_mapping(dim, bas_aa, bas_ab, measured_tpdm_aa, measured_tpdm_bb, measured_tpdm_ab):
    """
    Generate constraints such that the error matrix and the d2 matrices look like the measured matrices

    :param dim: spatial basis dimension
    :param measured_tpdm_aa: two-marginal of alpha-alpha spins
    :param measured_tpdm_bb: two-marginal of beta-beta spins
    :param measured_tpdm_ab: two-marginal of alpha-beta spins
    :return:
    """
    db = DualBasis()
    # first constrain the aa-matrix
    aa_dim = dim * (dim - 1) / 2
    ab_dim = dim **2

    # map the aa matrix to the measured_tpdm_aa
    for p, q, r, s in product(range(dim), repeat=4):
        if p < q and r < s and bas_aa[(p, q)] <= bas_aa[(r, s)]:
            dbe = DualBasisElement()

            # two elements of D2aa
            dbe.add_element('cckk_aa', (p, q, r, s), 0.5)
            dbe.add_element('cckk_aa', (r, s, p, q), 0.5)

            # four elements of the E2aa
            dbe.add_element('cckk_me_aa', (bas_aa[(p, q)] + aa_dim, bas_aa[(r, s)]), 0.25)
            dbe.add_element('cckk_me_aa', (bas_aa[(r, s)] + aa_dim, bas_aa[(p, q)]), 0.25)
            dbe.add_element('cckk_me_aa', (bas_aa[(p, q)], bas_aa[(r, s)] + aa_dim), 0.25)
            dbe.add_element('cckk_me_aa', (bas_aa[(r, s)], bas_aa[(p, q)] + aa_dim), 0.25)

            dbe.dual_scalar = measured_tpdm_aa[bas_aa[(p, q)], bas_aa[(r, s)]].real
            dbe.simplify()

            # construct the dbe for constraining the [0, 0] orthant to the idenity matrix
            dbe_identity_aa = DualBasisElement()
            if bas_aa[(p, q)] == bas_aa[(r, s)]:
                dbe_identity_aa.add_element('cckk_me_aa', (bas_aa[(p, q)], bas_aa[(r, s)]), 1.0)
                dbe_identity_aa.dual_scalar = 1.0
            else:
                dbe_identity_aa.add_element('cckk_me_aa', (bas_aa[(p, q)], bas_aa[(r, s)]), 0.5)
                dbe_identity_aa.add_element('cckk_me_aa', (bas_aa[(r, s)], bas_aa[(p, q)]), 0.5)
                dbe_identity_aa.dual_scalar = 0.0

            db += dbe
            db += dbe_identity_aa

    # map the bb matrix to the measured_tpdm_bb
    for p, q, r, s in product(range(dim), repeat=4):
        if p < q and r < s and bas_aa[(p, q)] <= bas_aa[(r, s)]:
            dbe = DualBasisElement()

            # two elements of D2bb
            dbe.add_element('cckk_bb', (p, q, r, s), 0.5)
            dbe.add_element('cckk_bb', (r, s, p, q), 0.5)

            # four elements of the E2bb
            dbe.add_element('cckk_me_bb', (bas_aa[(p, q)] + aa_dim, bas_aa[(r, s)]), 0.25)
            dbe.add_element('cckk_me_bb', (bas_aa[(r, s)] + aa_dim, bas_aa[(p, q)]), 0.25)
            dbe.add_element('cckk_me_bb', (bas_aa[(p, q)], bas_aa[(r, s)] + aa_dim), 0.25)
            dbe.add_element('cckk_me_bb', (bas_aa[(r, s)], bas_aa[(p, q)] + aa_dim), 0.25)

            dbe.dual_scalar = measured_tpdm_bb[bas_aa[(p, q)], bas_aa[(r, s)]].real
            dbe.simplify()

            # construct the dbe for constraining the [0, 0] orthant to the idenity matrix
            dbe_identity_bb = DualBasisElement()
            if bas_aa[(p, q)] == bas_aa[(r, s)]:
                dbe_identity_bb.add_element('cckk_me_bb', (bas_aa[(p, q)], bas_aa[(r, s)]), 1.0)
                dbe_identity_bb.dual_scalar = 1.0
            else:
                dbe_identity_bb.add_element('cckk_me_bb', (bas_aa[(p, q)], bas_aa[(r, s)]), 0.5)
                dbe_identity_bb.add_element('cckk_me_bb', (bas_aa[(r, s)], bas_aa[(p, q)]), 0.5)
                dbe_identity_bb.dual_scalar = 0.0

            db += dbe
            db += dbe_identity_bb

    # map the ab matrix to the measured_tpdm_ab
    for p, q, r, s in product(range(dim), repeat=4):
        if bas_ab[(p, q)] <= bas_ab[(r, s)]:
            dbe = DualBasisElement()

            # two elements of D2ab
            dbe.add_element('cckk_ab', (p, q, r, s), 0.5)
            dbe.add_element('cckk_ab', (r, s, p, q), 0.5)

            # four elements of the E2ab
            dbe.add_element('cckk_me_ab', (bas_ab[(p, q)] + ab_dim, bas_ab[(r, s)]), 0.25)
            dbe.add_element('cckk_me_ab', (bas_ab[(r, s)] + ab_dim, bas_ab[(p, q)]), 0.25)
            dbe.add_element('cckk_me_ab', (bas_ab[(p, q)], bas_ab[(r, s)] + ab_dim), 0.25)
            dbe.add_element('cckk_me_ab', (bas_ab[(r, s)], bas_ab[(p, q)] + ab_dim), 0.25)

            dbe.dual_scalar = measured_tpdm_ab[bas_ab[(p, q)], bas_ab[(r, s)]].real
            dbe.simplify()

            # construct the dbe for constraining the [0, 0] orthant to the idenity matrix
            dbe_identity_ab = DualBasisElement()
            if bas_ab[(p, q)] == bas_ab[(r, s)]:
                dbe_identity_ab.add_element('cckk_me_ab', (bas_ab[(p, q)], bas_ab[(r, s)]), 1.0)
                dbe_identity_ab.dual_scalar = 1.0
            else:
                dbe_identity_ab.add_element('cckk_me_ab', (bas_ab[(p, q)], bas_ab[(r, s)]), 0.5)
                dbe_identity_ab.add_element('cckk_me_ab', (bas_ab[(r, s)], bas_ab[(p, q)]), 0.5)
                dbe_identity_ab.dual_scalar = 0.0

            db += dbe
            db += dbe_identity_ab

    return db


def sz_adapted_linear_constraints(dim, Na, Nb, constraint_list, S=0, M=0):
    """
    Generate the dual basis for the v2-RDM program

    :param dim: rank of the spatial single-particle basis
    :param Na: Number of alpha electrons
    :param Nb: Number of beta electrons
    :param constraint_list:  List of strings indicating which constraints to make
    :return:
    """
    if Na != Nb and M != 0:
        raise TypeError("you gave me impossible quantum numbers")

    dual_basis = DualBasis()
    if 'cckk' in constraint_list:
        dual_basis += trace_d2_ab(dim, Na, Nb)
        dual_basis += s_representability_d2ab(dim, Na + Nb, M, S)

    # Including these would introduce linear independence.  Why?
    #     dual_basis += trace_d2_aa(dim, Na)
    #     dual_basis += trace_d2_bb(dim, Nb)

    if 'ck' in constraint_list:
        if Na > 1:
            dual_basis += d2aa_d1a_mapping(dim, Na)
        else:
            dual_basis += trace_d2_aa(dim, Na)
        if Nb > 1:
            dual_basis += d2bb_d1b_mapping(dim, Nb)
        else:
            dual_basis += trace_d2_bb(dim, Nb)

        dual_basis += d2ab_d1b_mapping(dim, Na)
        dual_basis += d2ab_d1a_mapping(dim, Nb)

        dual_basis += d1a_q1a_mapping(dim)
        dual_basis += d1b_q1b_mapping(dim)

        # dual_basis += d1a_d1b_mapping('ck_a', 'ck_b', dim)

        # this might not be needed if s_representability is enforced
        # if Na + Nb > 2:
        #     dual_basis += sz_representability(dim, M)

    if 'kkcc' in constraint_list:
        dual_basis += d2_q2_mapping(dim)

    if 'ckck' in constraint_list:
        dual_basis += d2_g2_mapping(dim)

    return dual_basis
