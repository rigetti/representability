"""
This module contains methods to find the closest positive semidefinite matrix with fixed trace
by the method in arXiv 1707.01022v1 and N.J. Higham, Linear Algebra and Its Applications 103, 103 (1998)
"""
from itertools import product
import numpy as np


@np.vectorize
def heaviside(x, bias=0):
    """
    Heaviside function Theta(x - bias)

    returns 1 if x >= bias else 0

    :param x:
    :param bias:
    :return:
    """
    indicator = 1 if x >= bias else 0
    return indicator


def higham_polynomial(eigenvalues, shift):
    """
    Calculate the higham_polynomial

    :param eigenvalues:
    :param shift:
    :return:
    """
    heaviside_indicator = np.asarray(heaviside(eigenvalues, bias=shift))
    return heaviside_indicator.T.dot(eigenvalues - shift)


def higham_root(eigenvalues, target_trace, epsilon=1.0E-15):
    """
    Find the root of f(sigma) = sum_{j}Theta(l_{i} - sigma)(l_{i} - sigma) = T

    :param eigenvalues: ordered list of eigenvalues from least to greatest
    :param target_trace: trace to maintain on new matrix
    :param epsilon: precision on bisection linesearch
    :return:
    """
    if target_trace < 0.0:
        raise ValueError("Target trace needs to be a non-negative number")

    # if eigenvalues[-1] <= 0 or eigenvalues[0] >= 0:
    #     raise ValueError("eigen spectrum is either all positive of all negative")

    # when we want the trace to be zero
    if np.isclose(target_trace, 0.0):
        return eigenvalues[-1]

    # find top sigma
    sigma = eigenvalues[-1]
    while higham_polynomial(eigenvalues, sigma) < target_trace:
        sigma -= eigenvalues[-1]

    sigma_low = sigma
    sigma_high = eigenvalues[-1]

    while sigma_high - sigma_low >= epsilon:
        midpoint = sigma_high - (sigma_high - sigma_low) / 2.0
        if higham_polynomial(eigenvalues, midpoint) < target_trace:
            sigma_high = midpoint
        else:
            sigma_low = midpoint

    return sigma_high


def map_to_matrix(mat):
    if mat.ndim != 4:
        raise TypeError("I only map rank-4 tensors to matices with symmetric support")
    dim = mat.shape[0]
    matform = np.zeros((dim**2, dim**2))
    for p, q, r, s in product(range(dim), repeat=4):
        assert np.isclose(mat[p, q, r, s].imag, 0.0)
        matform[p * dim + q, r * dim + s] = mat[p, q, r, s].real
    return matform


def map_to_tensor(mat):
    if mat.ndim != 2:
        raise TypeError("I only map matrices to rank-4 tensors with symmetric support")
    dim = int(np.sqrt(mat.shape[0]))
    tensor_form = np.zeros((dim, dim, dim, dim))
    for p, q, r, s in product(range(dim), repeat=4):
        tensor_form[p, q, r, s] = mat[p * dim + q, r * dim + s]
    return tensor_form


def fixed_trace_positive_projection(bmat, target_trace):
    """
    Perform the positive projection with fixed trace

    :param bmat:
    :param target_trace:
    :return:
    """
    bmat = np.asarray(bmat)
    if not np.allclose(bmat.imag, 0.0):
        raise TypeError("This code works with real matrices only")

    map_to_four_tensor = False
    if bmat.ndim == 4:
        bmat = map_to_matrix(bmat)
        map_to_four_tensor = True

    # symmeterize bmat
    if np.allclose(bmat - bmat.T, np.zeros_like(bmat)):
        bmat = 0.5 * (bmat + bmat.T)

    w, v = np.linalg.eigh(bmat)
    if np.all(w >= -1.0*float(1.0E-15)) and np.isclose(np.sum(w), target_trace):
        purified_matrix = bmat
    else:
        sigma = higham_root(w, target_trace)
        shifted_eigs = np.multiply(heaviside(w - sigma), (w - sigma))
        purified_matrix = np.zeros_like(bmat)
        for i in range(w.shape[0]):
            purified_matrix += shifted_eigs[i] * v[:, [i]].dot(v[:, [i]].T)

    if map_to_four_tensor:
        purified_matrix = map_to_tensor(purified_matrix)

    return purified_matrix
