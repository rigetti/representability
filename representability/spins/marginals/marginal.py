import sys
import numpy as np
from itertools import combinations, product
from pyquil.paulis import sI, sX, sY, sZ, PauliSum
from referenceqvm.unitary_generator import tensor_up
pauli_labels = ('I', 'X', 'Y', 'Z')
pauli_basis = {'I': sI, 'X': sX, 'Y': sY, 'Z': sZ}


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


class MarginalGenerator(Density):
    """
    Container and methods for constructing marginals of a density
    """
    def __init__(self, rho):
        """
        Constructor for the marginal object

        :param rho:
        """
        super(MarginalGenerator, self).__init__(rho)

    def construct_p_marginals(self, marginal_order):
        """
        Construct the p-marginal of the system

        :param marginal_order:
        :return:
        """
        qubit_indices = range(self.num_qubits)
        qubit_marginals = combinations(qubit_indices, marginal_order)
        # for each qubit_marginals
        # generate the m-qubit density matrix with indices in qubit marginals
        marginal_dictionary = {}
        for qubit_set in qubit_marginals:
            marginal_dictionary[qubit_set] = self.generate_marginal(qubit_set)

        return marginal_dictionary

    def generate_marginal(self, qubit_set):
        """
        Construct a marginal

        :param qubit_set:
        :return:
        """
        pauli_label_basis = list(product(pauli_labels, repeat=len(qubit_set)))
        marginal_rank = len(qubit_set)
        marginal = np.zeros((2 ** marginal_rank, 2 ** marginal_rank),
                            dtype=complex)

        # get set of matrices serving as the operator basis
        marginal_basis = {}
        for ops in pauli_label_basis:
            pauli_group_op = sI(0)
            for qubit_idx, p_op in enumerate(ops):
                pauli_group_op *= pauli_basis[p_op](qubit_idx)
            marginal_basis[ops] = tensor_up(PauliSum([pauli_group_op]),
                                            marginal_rank)

        for ops in pauli_label_basis:
            pauli_group_op = sI(0)
            for qubit_idx, p_op in zip(qubit_set, ops):
                pauli_group_op *= pauli_basis[p_op](qubit_idx)
            p_op_full_space = tensor_up(PauliSum([pauli_group_op]),
                                        self.num_qubits)
            p_op_full_space /= 2 ** marginal_rank
            # TODO: Normalize pauli coefficients...Did we do it properly?
            basis_coeff = np.trace(p_op_full_space.dot(self.rho))
            marginal += basis_coeff * marginal_basis[ops]

        return marginal






