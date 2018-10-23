"""Purify a 2-marginal by the unitary decomposition technique

0. Decompose tpdm into it's three components of the unitary group
1. set 2D(0) to be fixed by the trace
2. set 2D(1) to be fixed by a valid opdm contracted from the real one
3. reconstruct total tpdm
5. diagonalize and find exposed operators
6. decompose exposed operators into unitary subgroups
7. add the trace-less piece to the reconstructed D2 by solving system of equations
8. return purified tpdm
"""
import sys
from itertools import product
import numpy as np
from representability.purification.unitary_decomp import (coleman_decomposition,
                        mazziotti_opdm_purification)
from representability.purification.higham import fixed_trace_positive_projection
from representability.fermions.utils import wedge_product
from representability.purification.fermionic_marginal import (map_tpdm_to_opdm,
                        antisymmeterizer, map_tpdm_to_tqdm)
from representability.fermions.utils import (four_tensor2matrix,
                        matrix2four_tensor, check_antisymmetric_d2)


def coleman_projection_dq(tpdm, N, error=1.0E-6, disp=False):
    """
    Fixe up the tpdm trace and 1-particle piece and then iterate over the

    Coleman decomp of exposed operators

    :param tpdm:
    :param N:
    :param float error:
    :param Bool disp: optional argument for displaying updates
    :return:
    """
    if __debug__:
        check_antisymmetric_d2(tpdm)

    dim = tpdm.shape[0]
    _, _, d2_2 = coleman_decomposition(tpdm)
    # check if this thing I grabbed is actually an antisymmetric operator
    if __debug__:
        check_antisymmetric_d2(d2_2)
        assert np.isclose(np.einsum('ijij', d2_2), 0.0)

    eye_wedge_eye = wedge_product(np.eye(dim), np.eye(dim)).astype(complex)
    eye_wedge_eye_mat = four_tensor2matrix(eye_wedge_eye)

    # we know d2_0 by fixed trace from particle number
    trace_value = N * (N - 1)
    d2_0 = ((2 * trace_value) / (dim * (dim - 1))) * eye_wedge_eye

    # we know d2_1 should be (N - 1) * opdm
    # so grab the opdm first from the noisy tpdm and then purify
    opdm = map_tpdm_to_opdm(tpdm, N)
    # opdm_new = mazziotti_opdm_purification(opdm, N)
    opdm_new = fixed_trace_positive_projection(opdm, N)

    # first multiply by N - 1 get back to just contracted.  Then another
    # for adjusting
    one_one_tensor = ((N - 1)**2) * opdm_new
    one_one_eye = wedge_product(one_one_tensor, np.eye(dim))
    one_one_eye_mat = four_tensor2matrix(one_one_eye)

    d2_1 = (4 / (dim - 2)) * one_one_eye
    d2_1 -= ((4 * trace_value) / (dim * (dim - 2))) * eye_wedge_eye

    if __debug__:
        d2_1_trial = (4 / (dim - 2)) * wedge_product(
            opdm_new - (trace_value/dim) * np.eye(dim), np.eye(dim))
        assert np.allclose(d2_1_trial, d2_1)


    # we need to check a bunch of stuff about the reconstruct tpdm
    # a) does it have all the right symmetry
    # b) does it have the right trace and does it contract appropriately
    # c) does the d2_0 + d2_1 look like a contracted tpdm_partial fix

    tpdm_partial_fix = d2_0 + d2_1 + d2_2
    d2_matrix = four_tensor2matrix(tpdm_partial_fix)
    if __debug__:
        check_antisymmetric_d2(tpdm_partial_fix)
        # check conversion back to the tpdm_partial_fix
        np.testing.assert_allclose(matrix2four_tensor(d2_matrix), tpdm_partial_fix)

    # grabbing the 2-Q-RDM
    tqdm = map_tpdm_to_tqdm(tpdm_partial_fix, opdm_new)
    q2_matrix = four_tensor2matrix(tqdm)
    if __debug__:
        check_antisymmetric_d2(tqdm)

    residual = 10
    iter_max = 3000
    iter = 0
    while residual > error and iter < iter_max:
        tpdm_current_iter = matrix2four_tensor(d2_matrix)
        opdm = map_tpdm_to_opdm(tpdm_current_iter, N)
        one_wedge_eye_mat = four_tensor2matrix(wedge_product(opdm, np.eye(dim)))
        tqdm = map_tpdm_to_tqdm(tpdm_current_iter, opdm)
        q2_matrix = four_tensor2matrix(tqdm)

        if __debug__:
            tqdm_test = 2 * eye_wedge_eye - 4 * wedge_product(opdm, np.eye(dim)) + tpdm_current_iter
            np.testing.assert_allclose(tqdm_test, tqdm)


        # diagonalize both d2 and q2
        w, v = np.linalg.eigh(d2_matrix)
        wq, vq = np.linalg.eigh(q2_matrix)

        # grab the exposing operators for both matrices
        exposed_operator = []
        for ii in range(w.shape[0]):
            if w[ii] < float(-1.0E-13):
                if __debug__:
                    # checks to see if I'm sane?
                    assert v[:, [ii]].shape == (d2_matrix.shape[0], 1)
                    assert v[:, [ii]].dot(np.conj(v[:, [ii]]).T).shape == d2_matrix.shape
                    # check_antisymmetric_index(v[:, [ii]])

                # get the exposing operator antisymmeterize
                eo = v[:, [ii]].dot(np.conj(v[:, [ii]]).T)
                eo_22_tensor = matrix2four_tensor(eo)
                eo_22_tensor_antiy = antisymmeterizer(eo_22_tensor)

                if __debug__:
                    check_antisymmetric_d2(eo_22_tensor_antiy)  # redundant sanity check
                    # check if operator is hermetian
                    np.testing.assert_allclose(eo, np.conj(eo).T)

                # add the operator to my list
                exposed_operator.append(four_tensor2matrix(eo_22_tensor_antiy))

        exposed_operator_q = []
        for ii in range(wq.shape[0]):
            if wq[ii] < float(-1.0E-13):
                if __debug__:
                    # checks to see if I'm sane?
                    assert vq[:, [ii]].shape == (q2_matrix.shape[0], 1)
                    assert vq[:, [ii]].dot(np.conj(vq[:, [ii]]).T).shape == q2_matrix.shape
                    # check_antisymmetric_index(v[:, [ii]])

                # get the exposing operator antisymmeterize
                eo = vq[:, [ii]].dot(np.conj(vq[:, [ii]]).T)
                eo_22_tensor = matrix2four_tensor(eo)
                eo_22_tensor_antiy = antisymmeterizer(eo_22_tensor)

                if __debug__:
                    check_antisymmetric_d2(eo_22_tensor_antiy)  # redundant sanity check
                    # check if operator is hermetian
                    np.testing.assert_allclose(eo, np.conj(eo).T)

                # add the operator to my list
                exposed_operator_q.append(four_tensor2matrix(eo_22_tensor_antiy))


        # Now we want to find the coleman decomp of each of the exposing operators
        num_eo = len(exposed_operator)
        zero_contraction_eo = []
        for ii in range(num_eo):
            eo = matrix2four_tensor(exposed_operator[ii])
            eo_0, eo_1, eo_2 = coleman_decomposition(eo)
            zero_contraction_eo.append(four_tensor2matrix(eo_2))

            if __debug__:
                # check result is sane
                assert np.isclose(np.einsum('ijij', eo_0), np.einsum('ijij', eo))
                # check if it contracts to zero
                assert np.allclose(np.einsum('ikjk', eo_2), np.zeros_like(eo_2))
                # check if the einsum is correct
                assert np.allclose(np.einsum('ikjk', eo), np.einsum('ikjk', eo_0 + eo_1))
                # check if it reconstructs
                assert np.allclose(eo_0 + eo_1 + eo_2, eo)
                # check if it's hermetian
                assert np.allclose(zero_contraction_eo[-1], np.conj(zero_contraction_eo[-1]).T)
                # check if eo_2 is antisymmetric
                check_antisymmetric_d2(eo_2)

        num_eo_q = len(exposed_operator_q)
        zero_contraction_eo_q = []
        for ii in range(num_eo_q):
            eo = matrix2four_tensor(exposed_operator_q[ii])
            eo_0, eo_1, eo_2 = coleman_decomposition(eo)
            zero_contraction_eo_q.append(four_tensor2matrix(eo_2))

            if __debug__:
                # check result is sane
                assert np.isclose(np.einsum('ijij', eo_0), np.einsum('ijij', eo))
                # check if it contracts to zero
                assert np.allclose(np.einsum('ikjk', eo_2), np.zeros_like(eo_2))
                # check if the einsum is correct
                assert np.allclose(np.einsum('ikjk', eo), np.einsum('ikjk', eo_0 + eo_1))
                # check if it reconstructs
                assert np.allclose(eo_0 + eo_1 + eo_2, eo)
                # check if it's hermetian
                assert np.allclose(zero_contraction_eo_q[-1], np.conj(zero_contraction_eo_q[-1]).T)
                # check if eo_2 is antisymmetric
                check_antisymmetric_d2(eo_2)

        # set up system of equations to solve for alpha terms
        Amatrix = np.zeros((num_eo + num_eo_q, num_eo + num_eo_q))
        bvector = np.zeros((num_eo + num_eo_q, 1))
        for i, j in product(range(num_eo), repeat=2):
            # alpha coeffs
            Amatrix[i, j] = np.trace(
                exposed_operator[i].dot(zero_contraction_eo[j])).real
            # beta vector component for i-term
            if i == j:
                bvector[i, 0] = np.trace(
                    d2_matrix.dot(exposed_operator[i])).real

        for i, j in product(range(num_eo), range(num_eo_q)):
            # beta coeffs
            Amatrix[i, j + num_eo] = np.trace(
                exposed_operator[i].dot(zero_contraction_eo_q[j])).real

        # now fill in the rows of the Amatrix for the hole exposed operators
        for i, j in product(range(num_eo_q), range(num_eo)):
            # alpha coeffs with q-exposed operator
            Amatrix[i + num_eo, j] = np.trace(
                exposed_operator_q[i].dot(zero_contraction_eo[j]).real
            )
        for i, j in product(range(num_eo_q), repeat=2):
            # beta coeffs with q-exposed operator
            Amatrix[i + num_eo, j + num_eo] = np.trace(
                exposed_operator_q[i].dot(zero_contraction_eo_q[j]).real
            )
            if i == j:
                bvector[i + num_eo, 0] = np.trace(
                    d2_matrix.dot(exposed_operator_q[i])).real
                bvector[i + num_eo, 0] += 2 * np.trace(
                                              exposed_operator_q[i].dot(
                                                  eye_wedge_eye_mat)).real
                bvector[i + num_eo, 0] -= 4 * np.trace(
                                              exposed_operator_q[i].dot(
                                                  one_wedge_eye_mat)).real

        # alpha_beta = np.linalg.solve(Amatrix, -bvector)
        [alpha_beta, _, _, _] = np.linalg.lstsq(Amatrix, -bvector)
        if __debug__:
            assert np.allclose(np.dot(Amatrix, alpha_beta), -bvector)

        # update the 2-RDM matrix
        d2_new = d2_matrix.copy()
        for ii in range(num_eo):
            d2_new += alpha_beta[ii] * zero_contraction_eo[ii]
        for ii in range(num_eo_q):
            d2_new += alpha_beta[ii + num_eo] * zero_contraction_eo_q[ii]

        if __debug__:
            # check if the new d2_new is orthogonal to zero_contraction
            for ii in range(num_eo):
                assert np.isclose(np.trace(d2_new.dot(exposed_operator[ii])), 0.0)

        # check for conversion of the iterative method
        w, v = np.linalg.eigh(d2_new)
        residual = np.linalg.norm(w[w < 0])

        if disp:
            print("iter {:5.0f}\tdiff {:3.10e}\t".format(iter, np.linalg.norm(d2_new - d2_matrix)), end='')
            print("trace new {:3.10f}\terror {:3.10e}".format(np.trace(d2_new).real, residual))

        d2_matrix = d2_new
        iter += 1

    tpdm = matrix2four_tensor(d2_matrix)
    return tpdm


def unitary_subspace_purification_fixed_initial_trace(tpdm, N, error=1.0E-6, disp=False):
    """
    Fixe up the tpdm trace and 1-particle piece and then iterate over the

    Coleman decomp of exposed operators

    :param tpdm:
    :param N:
    :param float error:
    :param Bool disp: optional argument for displaying updates
    :return:
    """
    if __debug__:
        check_antisymmetric_d2(tpdm)

    dim = tpdm.shape[0]
    _, _, d2_2 = coleman_decomposition(tpdm)
    # check if this thing I grabbed is actually an antisymmetric operator
    if __debug__:
        check_antisymmetric_d2(d2_2)
        assert np.isclose(np.einsum('ijij', d2_2), 0.0)

    eye_wedge_eye = wedge_product(np.eye(dim), np.eye(dim)).astype(complex)

    # we know d2_0 by fixed trace from particle number
    trace_value = N * (N - 1)
    d2_0 = ((2 * trace_value) / (dim * (dim - 1))) * eye_wedge_eye

    # we know d2_1 should be (N - 1) * opdm
    # so grab the opdm first from the noisy tpdm and then purify
    opdm = map_tpdm_to_opdm(tpdm, N)
    # opdm_new = mazziotti_opdm_purification(opdm, N)
    opdm_new = fixed_trace_positive_projection(opdm, N)

    # first multiply by N - 1 get back to just contracted.  Then another
    # for adjusting
    one_one_tensor = ((N - 1)**2) * opdm_new
    one_one_eye = wedge_product(one_one_tensor, np.eye(dim))

    d2_1 = (4 / (dim - 2)) * one_one_eye
    d2_1 -= ((4 * trace_value) / (dim * (dim - 2))) * eye_wedge_eye

    if __debug__:
        d2_1_trial = (4 / (dim - 2)) * wedge_product(
            opdm_new - (trace_value/dim) * np.eye(dim), np.eye(dim))
        assert np.allclose(d2_1_trial, d2_1)

    # we need to check a bunch of stuff about the reconstruct tpdm
    # a) does it have all the right symmetry
    # b) does it have the right trace and does it contract appropriately
    # c) does the d2_0 + d2_1 look like a contracted tpdm_partial fix

    tpdm_partial_fix = d2_0 + d2_1 + d2_2
    d2_matrix = four_tensor2matrix(tpdm_partial_fix)
    if __debug__:
        check_antisymmetric_d2(tpdm_partial_fix)
        # check conversion back to the tpdm_partial_fix
        np.testing.assert_allclose(matrix2four_tensor(d2_matrix), tpdm_partial_fix)

    residual = 10
    iter_max = 3000
    iter = 0
    while residual > error and iter < iter_max:
        w, v = np.linalg.eigh(d2_matrix)
        exposed_operator = []
        for ii in range(w.shape[0]):
            if w[ii] < float(-1.0E-13):
                if __debug__:
                    # checks to see if I'm sane?
                    assert v[:, [ii]].shape == (d2_matrix.shape[0], 1)
                    assert v[:, [ii]].dot(np.conj(v[:, [ii]]).T).shape == d2_matrix.shape
                    # check_antisymmetric_index(v[:, [ii]])

                # get the exposing operator antisymmeterize
                eo = v[:, [ii]].dot(np.conj(v[:, [ii]]).T)
                eo_22_tensor = matrix2four_tensor(eo)
                eo_22_tensor_antiy = antisymmeterizer(eo_22_tensor)

                if __debug__:
                    check_antisymmetric_d2(eo_22_tensor_antiy)  # redundant sanity check
                    # check if operator is hermetian
                    np.testing.assert_allclose(eo, np.conj(eo).T)

                # add the operator to my list
                exposed_operator.append(four_tensor2matrix(eo_22_tensor_antiy))


        # Now we want to find the coleman decomp of each of the exposing operators
        num_eo = len(exposed_operator)
        zero_contraction_eo = []
        for ii in range(num_eo):
            eo = matrix2four_tensor(exposed_operator[ii])
            eo_0, eo_1, eo_2 = coleman_decomposition(eo)
            zero_contraction_eo.append(four_tensor2matrix(eo_2))

            if __debug__:
                # check result is sane
                assert np.isclose(np.einsum('ijij', eo_0), np.einsum('ijij', eo))
                # check if it contracts to zero
                assert np.allclose(np.einsum('ikjk', eo_2), np.zeros_like(eo_2))
                # check if the einsum is correct
                assert np.allclose(np.einsum('ikjk', eo), np.einsum('ikjk', eo_0 + eo_1))
                # check if it reconstructs
                assert np.allclose(eo_0 + eo_1 + eo_2, eo)
                # check if it's hermetian
                assert np.allclose(zero_contraction_eo[-1], np.conj(zero_contraction_eo[-1]).T)
                # check if eo_2 is antisymmetric
                check_antisymmetric_d2(eo_2)

        # set up system of equations to solve for alpha terms
        Amatrix = np.zeros((num_eo, num_eo))
        bvector = np.zeros((num_eo, 1))
        for i, j in product(range(num_eo), repeat=2):
            Amatrix[i, j] = np.trace(exposed_operator[i].dot(zero_contraction_eo[j])).real
            if i == j:
                bvector[i, 0] = np.trace(d2_matrix.dot(exposed_operator[i])).real

        # alpha = np.linalg.solve(Amatrix, -bvector)
        [alpha, _, _, _] = np.linalg.lstsq(Amatrix, -bvector)
        if __debug__:
            assert np.allclose(np.dot(Amatrix, alpha), -bvector)

        # update the 2-RDM matrix
        d2_new = d2_matrix.copy()
        for ii in range(num_eo):
            d2_new += alpha[ii] * zero_contraction_eo[ii]

        if __debug__:
            # check if the new d2_new is orthogonal to zero_contraction
            for ii in range(num_eo):
                assert np.isclose(np.trace(d2_new.dot(exposed_operator[ii])), 0.0)

        # check for conversion of the iterative method
        w, v = np.linalg.eigh(d2_new)
        residual = np.linalg.norm(w[w < 0])

        if disp:
            print("iter {:5.0f}\tdiff {:3.10e}\t".format(iter, np.linalg.norm(d2_new - d2_matrix)), end='')
            print("trace new {:3.10f}\terror {:3.10e}".format(np.trace(d2_new).real, residual))

        d2_matrix = d2_new
        iter += 1

    tpdm = matrix2four_tensor(d2_matrix)
    return tpdm


def check_antisymmetric_index(v):
    """
    Check if we are antisymmetric in the indices of v

    :param v:
    :return:
    """
    dim = int(np.sqrt(v.shape[0]))
    for p, q in product(range(dim), repeat=2):
        if not np.isclose(v[p * dim + q], -1 * v[q * dim + p]):
            raise ValueError("my vector is not antisymmetric in its indices")


