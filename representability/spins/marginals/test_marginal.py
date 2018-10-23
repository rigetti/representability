"""
Test the marginal.py module
"""
import numpy as np
from representability.spins.marginals.marginal import MarginalGenerator
from pyquil.paulis import sI, sX, sY, sZ
from referenceqvm.unitary_generator import tensor_up


def test_marginal():
    """
    Generate ground state of heinsenberg spin model and general
    all 1-marginals

    Heisenberg spin model is

    H = sum_{<ij>} X_{i}X_{j} + Y_{i}Y_{j} + Z_{i}Z_{j}

    :return:
    """
    # generate state for Heisenberg spin-model
    qubits = 4
    hamiltonian = sI(0) * 0.0
    for ii in range(qubits):
        hamiltonian += sX(ii) * sX((ii + 1) % qubits)
        hamiltonian += sY(ii) * sY((ii + 1) % qubits)
        hamiltonian += sZ(ii) * sZ((ii + 1) % qubits)

    hamiltonian_matrix = tensor_up(hamiltonian, qubits)
    w, v = np.linalg.eigh(hamiltonian_matrix)
    rho = v[:, [0]].dot(np.conj(v[:, [0]]).T)
    mg = MarginalGenerator(rho)
    marginals = mg.construct_p_marginals(2)
    for marginal_id, marginal in marginals.items():
        assert np.isclose(marginal.trace(), 1.0)


def test_one_marginal():
    """
    Generate ground state of heinsenberg spin model and general
    all 1-marginals

    Heisenberg spin model is

    H = sum_{<ij>} X_{i}X_{j} + Y_{i}Y_{j} + Z_{i}Z_{j}

    :return:
    """
    # generate state for Heisenberg spin-model
    qubits = 2
    hamiltonian = sI(0) * 0.0
    for ii in range(qubits):
        hamiltonian += sX(ii) * sX((ii + 1) % qubits)
        hamiltonian += sY(ii) * sY((ii + 1) % qubits)
        hamiltonian += sZ(ii) * sZ((ii + 1) % qubits)

    hamiltonian_matrix = tensor_up(hamiltonian, qubits)
    w, v = np.linalg.eigh(hamiltonian_matrix)
    rho = v[:, [0]].dot(np.conj(v[:, [0]]).T)
    mg = MarginalGenerator(rho)
    marginals = mg.construct_p_marginals(1)
    for marginal_id, marginal in marginals.items():
        print(marginal_id)
        print(marginal)
