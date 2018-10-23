"""Unitary decomposition of antisymmetric Hermetian matrices and purification
algorithms based off of the decomp"""
import sys
import numpy as np
from representability.fermions.utils import wedge_product


def coleman_decomposition(two_two_tensor):
    """
    Perform unitary decomposition of a (2, 2)-tensor

    For more detail references and derivations see references:

    1. Phys. Rev. E. 65 026704
    2. Int. J. Quant. Chem. XVII 127901307 (1980)

    :param two_two_tensor: 4-tensor representing a (2, 2)-tensor
    :return:
    """
    if not np.isclose(np.ndim(two_two_tensor), 4):
        raise TypeError("coleman decomposition requires a (2,2) tensor")

    dim = two_two_tensor.shape[0]
    trace_value = np.einsum('ijij', two_two_tensor)

    one_one_tensor = np.einsum('ikjk', two_two_tensor)
    one_one_eye = wedge_product(one_one_tensor, np.eye(dim)).astype(complex)
    eye_wedge_eye = wedge_product(np.eye(dim), np.eye(dim)).astype(complex)

    zero_carrier = ((2 * trace_value) / (dim * (dim - 1))) * eye_wedge_eye

    one_carrier = (4 / (dim - 2)) * one_one_eye
    one_carrier -= ((4 * trace_value) / (dim * (dim - 2))) * eye_wedge_eye

    two_carrier = two_two_tensor - (4 / (dim - 2)) * one_one_eye
    two_carrier += ((2 * trace_value) / ((dim - 1) * (dim - 2))) * eye_wedge_eye

    return zero_carrier.real, one_carrier.real, two_carrier.real


def mazziotti_opdm_purification(opdm, trace_target):
    """
    Mazziotti purification of the 1-RDM in Phys. Rev. E. 65 026704

    i) set all negative eigenvalues to zero
    ii) correct the trace by decreasing the HOMO
    iii) set all eigenvalues >1 to 1
    iv) correct the trace by increasing the LUMO

    :param opdm: one-particle density matrix
    :param trace_target: true trace of the opdm
    :return: purified opdm
    """
    if not isinstance(opdm, np.ndarray):
        raise TypeError("opdm must be a numpy matrix")

    if np.ndim(opdm) != 2:
        raise TypeError("opdm must be a (1, 1)-tensor")

    dim = int(opdm.shape[0])
    [eigvals, eigvecs] = np.linalg.eigh(opdm)
    eigvals = eigvals.real
    if (1 >= eigvals).all and (eigvals >= 0).all() and np.isclose(np.sum(eigvals), trace_target):
        return opdm

    # step (i)
    for i in range(eigvals.shape[0]):
        if eigvals[i] < 0:
            eigvals[i] = 0

    # step (ii)
    # if we have more electrons than required
    if np.sum(eigvals) > trace_target:
        eigvals = decrease_homo_to_target(eigvals, trace_target)
    else:
        # we have less electrons. Scale to one and then make_occs_rep
        eigvals *= trace_target / np.sum(eigvals).real

    # step (iii)
    # set all eigs > 1 to 1
    for i in range(eigvals.shape[0]):
        if eigvals[i] > 1:
            eigvals[i] = 1

    # step (iv)
    # increase the LUMO such that the trace is preserved
    if np.sum(eigvals) < trace_target:
        eigvals = increase_lumo_to_target(eigvals, trace_target)
    elif np.isclose(np.sum(eigvals), trace_target):
        pass
    else:
        raise RuntimeError("I think the impossible has happened or Nick dun goofed")

    opdm_new = eigvecs.dot(np.diag(eigvals)).dot(np.conj(eigvecs.T))
    return opdm_new


def increase_lumo_to_target(eigvals, trace_target):
    """
    Increase the eignvalues to the target

    :param eigvals:
    :param trace_target:
    :return:
    """
    if np.sum(eigvals) >= trace_target:
        raise ValueError("sum of eigenvalues must be less than the target")

    # find the lumo
    num_occ_index = 0
    lumo = -1
    while num_occ_index < len(eigvals):
        if np.isclose(eigvals[num_occ_index], 0.0):
            num_occ_index += 1
        else:
            lumo = num_occ_index - 1
            break

    while lumo >= 0:
        # get amo
        trace_diff = trace_target - np.sum(eigvals)
        if trace_diff > 1:
            eigvals[lumo] = 1
            lumo -= 1
        else:
            eigvals[lumo] += trace_diff
            break

    return eigvals


def decrease_homo_to_target(eigvals, trace_target):
    """
    Make occupation numbers representable

    This functions expects the sum of eigs to be equal to or greater than the
    target trace

    :param eigvals: numpy array or list of eigenvalues
    :return:
    """
    if np.sum(eigvals) < trace_target:
        raise ValueError("make_occs_representable expects trace to be >= target")

    num_occ_index = 0
    num_eigvals = len(eigvals)
    while num_occ_index < num_eigvals:
        total_trace_diff = np.sum(eigvals) - trace_target
        if total_trace_diff > eigvals[num_occ_index]:
            eigvals[num_occ_index] = 0
            num_occ_index += 1
        else:
            eigvals[num_occ_index] -= total_trace_diff.real
            break

    return eigvals

