import pytest
from itertools import product
import numpy as np
from pyquil.paulis import PauliTerm, PauliSum
from referenceqvm.unitary_generator import tensor_up
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner
from openfermion.ops import FermionOperator
from openfermion.utils import (map_two_pdm_to_two_hole_dm, map_two_pdm_to_particle_hole_dm)
from openfermionpsi4 import run_psi4
from forestopenfermion.pyquil_connector import qubitop_to_pyquilpauli
from representability.fermions.density.spin_density import SpinOrbitalDensity
from representability.fermions.utils import get_molecule_openfermion


# probably want to upgrade this with yield fixture.  This will need to be an object
def system():
    print('Running System Setup')
    basis = 'sto-3g'
    multiplicity = 1
    charge = 1
    geometry = [('He', [0.0, 0.0, 0.0]), ('H', [0, 0, 0.740848149])]
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    # Run Psi4.
    molecule = run_psi4(molecule,
                        run_scf=True,
                        run_mp2=False,
                        run_cisd=False,
                        run_ccsd=False,
                        run_fci=True,
                        delete_input=False)

    molecule, gs_wf, n_density, eigen_val = get_molecule_openfermion(molecule, eigen_index=2)
    rdm_generator = SpinOrbitalDensity(n_density, molecule.n_qubits)
    transform = jordan_wigner
    return n_density, rdm_generator, transform, molecule


def test_construct_opdm():
    """
    Test the construction of one-particle density matrix

    <psi|a_{p}^{\dagger}a_{q}|psi>
    """
    rho, rdm_generator, transform, molecule = system()
    opdm = np.zeros((molecule.n_qubits, molecule.n_qubits), dtype=complex)
    for p, q in product(range(molecule.n_qubits), repeat=2):
        pauli_proj_op = transform(FermionOperator(((p, 1), (q, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        opdm_element = np.trace(lifted_op.dot(rho))
        opdm[p, q] = opdm_element

    assert np.allclose(molecule.fci_one_rdm, opdm)

    opdm_test = rdm_generator._tensor_construct(2, [-1, 1])
    assert np.allclose(opdm_test, molecule.fci_one_rdm)


@pytest.mark.skip(reason='very slow and redundant test. Run if curious')
def test_construct_ohdm():
    """
    Test the construction of the one-hole density matrix

    <psi|a_{p}a_{q}^{\dagger}|psi>
    """
    rho, rdm_generator, transform, molecule = system()
    dim = molecule.n_qubits
    ohdm = np.zeros((dim, dim), dtype=complex)
    for p, q in product(range(dim), repeat=2):
        pauli_proj_op = transform(FermionOperator(((p, 0), (q, 1))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        ohdm_element = np.trace(lifted_op.dot(rho))
        ohdm[p, q] = ohdm_element

    ohdm_true = np.eye(4) - molecule.fci_one_rdm
    assert np.allclose(ohdm_true, ohdm)
    ohdm_test = rdm_generator._tensor_construct(2, [1, -1])
    assert np.allclose(ohdm_test, ohdm_true)


@pytest.mark.skip(reason='very slow and redundant test. Run if curious')
def test_construct_tpdm():
    """
    Construct the two-particle density matrix

    <psi|a_{p}^{\dagger}a_{q}^{\dagger}a_{s}a_{r}|psi>
    """
    rho, rdm_generator, transform, molecule = system()
    dim = molecule.n_qubits
    tpdm = np.zeros((dim, dim, dim, dim), dtype=complex)
    for p, q, r, s in product(range(dim), repeat=4):
        pauli_proj_op = transform(FermionOperator(((p, 1), (q, 1), (s, 0), (r, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        tpdm_element = np.trace(lifted_op.dot(rho))
        tpdm[p, q, r, s] = tpdm_element

    tpdm_test = rdm_generator._tensor_construct(4, [-1, -1, 1, 1])
    assert np.allclose(tpdm_test, tpdm)
    assert np.allclose(tpdm_test, np.einsum('ijkl->ijlk', molecule.fci_two_rdm))
    assert np.allclose(tpdm, np.einsum('ijkl->ijlk', molecule.fci_two_rdm))


@pytest.mark.skip(reason='very slow and redundant test. Run if curious')
def test_construct_thdm():
    """
    Construct the two-hole density matrix

    <psi|a_{p}a_{q}a_{s}^{\dagger}a_{r}^{\dagger}|psi>
    """
    rho, rdm_generator, transform, molecule = system()
    dim = molecule.n_qubits
    thdm = np.zeros((dim, dim, dim, dim), dtype=complex)
    for p, q, r, s in product(range(dim), repeat=4):
        pauli_proj_op = transform(FermionOperator(((p, 0), (q, 0), (s, 1), (r, 1))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        thdm_element = np.trace(lifted_op.dot(rho))
        thdm[p, q, r, s] = thdm_element

    thdm_test = rdm_generator._tensor_construct(4, [1, 1, -1, -1])
    assert np.allclose(thdm_test, thdm)

    thdm_true = map_two_pdm_to_two_hole_dm(molecule.fci_two_rdm, molecule.fci_one_rdm)
    thdm_true = np.einsum('ijkl->ijlk', thdm_true)
    assert np.allclose(thdm_test, thdm_true)


def test_construct_phdm():
    """
    Construct the particle-hole density matrix

    <psi|a_{p}^{\dagger}a_{q}a_{s}^{\dagger}a_{r}|psi>
    """
    rho, rdm_generator, transform, molecule = system()
    dim = molecule.n_qubits
    phdm = np.zeros((dim, dim, dim, dim), dtype=complex)
    for p, q, r, s in product(range(dim), repeat=4):
        pauli_proj_op = transform(FermionOperator(((p, 1), (q, 0), (s, 1), (r, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        phdm_element = np.trace(lifted_op.dot(rho))
        phdm[p, q, r, s] = phdm_element

    phdm_test = rdm_generator._tensor_construct(4, [-1, 1, -1, 1])
    assert np.allclose(phdm_test, phdm)

    phdm_true = map_two_pdm_to_particle_hole_dm(molecule.fci_two_rdm, molecule.fci_one_rdm)
    phdm_true = np.einsum('ijkl->ijlk', phdm_true)
    assert np.allclose(phdm_true, phdm)


if __name__ == "__main__":
    # test_construct_ohdm()
    # test_construct_opdm()
    test_construct_tpdm()
    test_construct_thdm()
    test_construct_phdm()
