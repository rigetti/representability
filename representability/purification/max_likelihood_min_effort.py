"""Maximum Likelihood Minimum Effort Routines.
This routine courtesy of J. R. McClean"""
import numpy
import scipy
from functools import reduce


def project_density_matrix(rho):
    """Project a density matrix to the closest positive semi-definite matrix with
        trace 1.  Follows the algorithm in PhysRevLett.108.070502
        Args:
            rho: Numpy array containing the density matrix with dimension (N, N)
        Returns:
            rho_projected: The closest positive semi-definite trace 1 matrix to rho.
    """

    # Rescale to trace 1 if the matrix is not already
    rho_impure = rho / numpy.trace(rho)

    dimension = rho_impure.shape[0]  # the dimension of the Hilbert space
    [eigvals, eigvecs] = scipy.linalg.eigh(rho_impure)

    # If matrix is already trace one PSD, we are done
    if numpy.min(eigvals) >= 0:
        return rho_impure

    # Otherwise, continue finding closest trace one, PSD matrix
    eigvals = list(eigvals)
    eigvals.reverse()
    eigvals_new = [0.0] * len(eigvals)

    i = dimension
    accumulator = 0.0  # Accumulator
    while eigvals[i - 1] + accumulator / float(i) < 0:
        accumulator += eigvals[i - 1]
        i -= 1
    for j in range(i):
        eigvals_new[j] = eigvals[j] + accumulator / float(i)
    eigvals_new.reverse()

    # Reconstruct the matrix
    rho_projected = reduce(numpy.dot, (eigvecs,
                                   numpy.diag(eigvals_new),
                                   numpy.conj(eigvecs.T)))

    return rho_projected
