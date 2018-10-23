"""
Purify a Fermionic marginal via the iterative procedure in arXiv:1707.01022

Note: the antisymmetry of the 2-RDM is not constrained in the spin-orbital code below
Solution: Utilize the spin-adapted marginals over antisymmetric basis functions to remove the explicit inclusion of
antisymmetry
"""
import numpy as np
from itertools import product
from representability.purification.higham import fixed_trace_positive_projection, map_to_matrix


def symmeterize_matrix(mat):
    """
    make a matrix symmetric

    :param matrix:
    :return:
    """
    if mat.ndim != 2:
        raise TypeError("mat is not a matrix. dim is {}".format(mat.ndim))
    return 0.5 * (mat + mat.T)


def symmeterize_four_tensor(mat):
    """
    make a matrix symmetric

    :param matrix:
    :return:
    """
    if mat.ndim != 4:
        raise TypeError("mat is not a 4-tensor. dim is {}".format(mat.ndim))
    dim = mat.shape[0]
    for p, q, r, s in product(range(dim), repeat=4):
        if p * dim + q <= r * dim + s:
            mat[p, q, r, s] = 0.5 * (mat[p, q, r, s] + mat[r, s, p, q])
            mat[r, s, p, q] = mat[p, q, r, s]
    return mat


def map_tpdm_to_tqdm(tpdm, opdm):
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
        tqdm[r, s, p, q] = tpdm[p, q, r, s] - term1 - term2 - term3

    return tqdm


def map_tqdm_to_tpdm(tqdm, opdm):
    """
    map the two-hole density matrix to the two-particle density matrix

    :param tpdm: rank-4 tensor representing the 2-HDM
    :param opdm: rank-2 tensor representing the 1-RDM
    :returns: rank-4 tensor representing the 2-RDM
    :rtype: ndarray
    """
    sm_dim = opdm.shape[0]
    krond = np.eye(sm_dim)
    tpdm = np.zeros((sm_dim, sm_dim, sm_dim, sm_dim), dtype=complex)
    for p, q, r, s in product(range(sm_dim), repeat=4):
        term1 = opdm[p, r]*krond[q, s] + opdm[q, s]*krond[p, r]
        term2 = -1*(opdm[p, s]*krond[r, q] + opdm[q, r]*krond[s, p])
        term3 = krond[s, p]*krond[r, q] - krond[q, s]*krond[r, p]
        tpdm[p, q, r, s] = tqdm[r, s, p, q] + term1 + term2 + term3

    return tpdm


def map_tpdm_to_phdm(tpdm, opdm):
    """
    map the two-particle density matrix to the particle-hole density matrix

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
        tgdm[p, s, r, q] = term1.real - tpdm[p, q, r, s].real
    return tgdm


def map_phdm_to_tpdm(phdm, opdm):
    """
    map the particle-hole marginal to the two-particle marginal

    :param phdm: rank-4 tensor representing the G-matrix
    :param opdm: rank-2 tensor representing the 1-RDM
    :returns: rank-4 tensor representing the P-H-DM
    :rtype: ndarray
    """
    sm_dim = opdm.shape[0]
    krond = np.eye(sm_dim)
    tpdm = np.zeros_like(phdm)
    for p, q, r, s in product(range(sm_dim), repeat=4):
        term1 = opdm[p, r]*krond[q, s]
        tpdm[p, q, r, s] = term1 - phdm[p, s, r, q]
    return tpdm


def map_tpdm_to_opdm(tpdm, normalization):
    """
    map the two-particle marginal to the one-particle marginal

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


def map_phdm_to_opdm(phdm, N, spin_orbitals):
    return np.einsum('ijkj', phdm)/ (spin_orbitals - N + 1)


def map_opdm_to_oqdm(opdm):
    krond = np.eye(opdm.shape[0])
    return krond - opdm


def map_oqdm_to_opdm(oqdm):
    krond = np.eye(oqdm.shape[0])
    return krond - oqdm


def map_tqdm_to_oqdm(tqdm, normalization):
    return map_tpdm_to_opdm(tqdm, normalization)


def antisymmeterizer(marginal):
    new_marginal = np.zeros_like(marginal)
    m = marginal.shape[0]
    for p, q, r, s in product(range(m), repeat=4):
        if p < q and r < s:
            element_average = 0.25 * (marginal[p, q, r, s] + -1*marginal[q, p, r, s] +
                                      -1*marginal[p, q, s, r] + marginal[q, p, s, r])
            new_marginal[p, q, r, s] = element_average
            new_marginal[q, p, r, s] = -1 * element_average
            new_marginal[p, q, s, r] = -1 * element_average
            new_marginal[q, p, s, r] = element_average

    return new_marginal


def purification_step(marginal, normalization, antisymmetric=False):
    marginal = symmeterize_four_tensor(marginal)
    w, v = np.linalg.eigh(map_to_matrix(marginal))
    # eigen error is absolute value of largest negative or zero
    marginal_eig_error = np.abs(w[0]) if w[0] < 0 else 0
    marginal_trace_error = np.sum(w) - normalization

    marginal = fixed_trace_positive_projection(marginal, normalization)
    return marginal, marginal_eig_error, marginal_trace_error


def purify_marginal(tpdm, N, M, epsilon=1.0E-6, max_iter=500, disp=False):
    """
    Purify the two-particle marginal by iteratively purifying oqdm and phdm

    :param ndarray tpdm: two-particle marginal as a rank-4 tensor
    :param Int N: Total number of electrons
    :param Int M: Total number of spin-orbital basis functions
    :param Float epsilon: convergence criteria for purification iteration.  Bounds the Frobenius norm of the difference
                    of 2-particle marginals between purification cycles.
    :param Int max_iter: maximum number of purification cycles
    :param Bool disp: Display iteration information
    :return: purified two-particle marginal
    :rtype: np.ndarray
    """
    # number of holes in my system
    eta = M - N

    converged = False
    iter_number = 0

    if disp:
        print("\t{}\t{}\t{}\t{}".format("Iter Numbers", "Residual", "Trace Errors", "Eigen Errors"))

    while not converged and iter_number < max_iter:
        # fix tpdm to be positive and given trace and contract to opdm
        tpdm, tpdm_eig_error, tpdm_trace_error = purification_step(tpdm, N * (N - 1))
        opdm = map_tpdm_to_opdm(tpdm, N)
        assert np.isclose(np.trace(opdm), N)

        # tpdm |-> tqdm, purifiy, tqdm |-> tqdm
        # note: we need to contract to the Q-matrix to the 1-RDM so the iterative procedure doesn't blow up
        tqdm = map_tpdm_to_tqdm(tpdm, opdm)
        tqdm, tqdm_eig_error, tqdm_trace_error = purification_step(tqdm, eta * (eta - 1))
        oqdm = map_tqdm_to_oqdm(tqdm, eta)
        opdm = map_oqdm_to_opdm(oqdm)
        tpdm = map_tqdm_to_tpdm(tqdm, opdm)

        # map tpdm to phdm and purify
        # Note: we need to contract the G-matrix to the 1-RDM so the iterative procedure doesn't blow up
        phdm = map_tpdm_to_phdm(tpdm, opdm)
        phdm, phdm_eig_error, phdm_trace_error = purification_step(phdm, N * (eta + 1))
        opdm = map_phdm_to_opdm(phdm, N, M)
        tpdm = map_phdm_to_tpdm(phdm, opdm)

        errors = [tpdm_trace_error, tqdm_trace_error, phdm_trace_error, tpdm_eig_error, tqdm_eig_error, phdm_eig_error]

        residual = sum(map(lambda x: x**2, errors[:3])) + max(errors[3:])

        if disp:
            print("\t{}\t\t{:2.5E}\t{:2.5E}\t{:2.5E}".format(iter_number, residual, np.linalg.norm(errors[:3]), np.linalg.norm(errors[3:])))

        if residual < epsilon:
            converged = True
        iter_number += 1

    return tpdm

