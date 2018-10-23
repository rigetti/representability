import pytest
from itertools import product
import numpy as np
from referenceqvm.unitary_generator import tensor_up
from pyquil.paulis import PauliTerm, PauliSum
from forestopenfermion.pyquil_connector import qubitop_to_pyquilpauli
from representability.fermions.density.symm_sz_density import SymmOrbitalDensity
from representability.fermions.utils import get_molecule_openfermion
from representability.config import *
from openfermion.transforms import jordan_wigner
from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator
from openfermionpsi4 import run_psi4


# probably want to upgrade this with yield fixture.  This will need to be an object
def system():
    print('Running System Setup')
    basis = 'sto-3g'
    multiplicity = 1
    charge = 1
    geometry = [('He', [0.0, 0.0, 0.0]), ('H', [0, 0, 0.75])]
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
    rdm_generator = SymmOrbitalDensity(n_density, molecule.n_qubits)
    transform = jordan_wigner
    return n_density, rdm_generator, transform, molecule


def test_construct_opdm():
    """
    Test the construction of one-particle density matrix

    <psi|a_{p}^{\dagger}a_{q}|psi>
    """
    rho, rdm_generator, transform, molecule = system()
    dim = molecule.n_qubits
    opdm_a = np.zeros((int(dim/2), int(dim/2)), dtype=complex)
    opdm_b = np.zeros((int(dim/2), int(dim/2)), dtype=complex)

    for p, q in product(range(int(dim/2)), repeat=2):
        pauli_proj_op = transform(FermionOperator(((2 * p, 1), (2 * q, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        opdm_element = np.trace(lifted_op.dot(rho))
        opdm_a[p, q] = opdm_element

        pauli_proj_op = transform(FermionOperator(((2 * p + 1, 1), (2 * q + 1, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        opdm_element = np.trace(lifted_op.dot(rho))
        opdm_b[p, q] = opdm_element

    opdm_a_test, opdm_b_test = rdm_generator.construct_opdm()
    assert np.allclose(opdm_b, opdm_b_test)
    assert np.allclose(opdm_a, opdm_a_test)
    opdm_b_test = rdm_generator._tensor_construct(2, [-1, 1], [1, 1])
    assert np.allclose(opdm_b_test, opdm_b)
    opdm_a_test = rdm_generator._tensor_construct(2, [-1, 1], [0, 0])
    assert np.allclose(opdm_a_test, opdm_a)


def test_construct_ohdm():
    """
    Test the construction of the one-hole density matrix

    <psi|a_{p}a_{q}^{\dagger}|psi>
    """
    rho, rdm_generator, transform, molecule = system()
    dim = molecule.n_qubits

    ohdm_a = np.zeros((int(dim/2), int(dim/2)), dtype=complex)
    ohdm_b = np.zeros((int(dim/2), int(dim/2)), dtype=complex)
    for p, q in product(range(int(dim/2)), repeat=2):
        pauli_proj_op = transform(FermionOperator(((2 * p, 0), (2 * q, 1))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        ohdm_element = np.trace(lifted_op.dot(rho))
        ohdm_a[p, q] = ohdm_element

        pauli_proj_op = transform(FermionOperator(((2 * p + 1, 0), (2 * q + 1, 1))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        ohdm_element = np.trace(lifted_op.dot(rho))
        ohdm_b[p, q] = ohdm_element


    ohdm_a_test, ohdm_b_test = rdm_generator.construct_ohdm()
    assert np.allclose(ohdm_a_test, ohdm_a)
    assert np.allclose(ohdm_b_test, ohdm_b)

    ohdm_a_test = rdm_generator._tensor_construct(2, [1, -1], [0, 0])
    assert np.allclose(ohdm_a_test, ohdm_a)
    ohdm_b_test = rdm_generator._tensor_construct(2, [1, -1], [1, 1])
    assert np.allclose(ohdm_b_test, ohdm_b)


@pytest.mark.skip(reason='very slow and redundant test. Run if curious')
def test_construct_tpdm():
    """
    Construct the two-particle density matrix

    <psi|a_{p}^{\dagger}a_{q}^{\dagger}a_{s}a_{r}|psi>
    """
    rho, rdm_generator, transform, molecule = system()
    dim = molecule.n_qubits

    tpdm_aa = np.zeros((int(dim/2), int(dim/2), int(dim/2), int(dim/2)),
                       dtype=complex)
    tpdm_bb = np.zeros((int(dim/2), int(dim/2), int(dim/2), int(dim/2)),
                       dtype=complex)
    tpdm_ab = np.zeros((int(dim/2), int(dim/2), int(dim/2), int(dim/2)),
                       dtype=complex)
    tpdm_ba = np.zeros((int(dim/2), int(dim/2), int(dim/2), int(dim/2)),
                       dtype=complex)
    for p, q, r, s in product(range(int(dim/2)), repeat=4):
        # pauli_proj_op = transform.product_ops([2 * p + 1, 2 * q + 1,
        #                                        2 * s + 1, 2 * r + 1], [-1, -1, 1, 1])
        pauli_proj_op = transform(FermionOperator(((2 * p + 1, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r + 1, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        tpdm_element = np.trace(lifted_op.dot(rho))
        tpdm_bb[p, q, r, s] = tpdm_element

        # pauli_proj_op = transform.product_ops([2 * p, 2 * q,
        #                                        2 * s, 2 * r], [-1, -1, 1, 1])
        pauli_proj_op = transform(FermionOperator(((2 * p, 1), (2 * q, 1), (2 * s, 0), (2 * r, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        tpdm_element = np.trace(lifted_op.dot(rho))
        tpdm_aa[p, q, r, s] = tpdm_element

        # pauli_proj_op = transform.product_ops([2 * p, 2 * q + 1,
        #                                        2 * s + 1, 2 * r], [-1, -1, 1, 1])
        pauli_proj_op = transform(FermionOperator(((2 * p, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        tpdm_element = np.trace(lifted_op.dot(rho))
        tpdm_ab[p, q, r, s] = tpdm_element

        # pauli_proj_op = transform.product_ops([2 * p + 1, 2 * q,
        #                                        2 * s, 2 * r + 1], [-1, -1, 1, 1])
        pauli_proj_op = transform(FermionOperator(((2 * p + 1, 1), (2 * q, 1), (2 * s, 0), (2 * r + 1, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        tpdm_element = np.trace(lifted_op.dot(rho))
        tpdm_ba[p, q, r, s] = tpdm_element

    tpdm_aa_true, tpdm_bb_true, tpdm_ab_true, tpdm_ba_true = rdm_generator.construct_tpdm()
    assert np.allclose(tpdm_aa, tpdm_aa_true)
    assert np.allclose(tpdm_bb, tpdm_bb_true)
    assert np.allclose(tpdm_ab, tpdm_ab_true)
    assert np.allclose(tpdm_ba, tpdm_ba_true)


@pytest.mark.skip(reason='very slow and redundant test. Run if curious')
def test_construct_thdm():
    """
    Construct the two-hole density matrix

    <psi|a_{p}a_{q}a_{s}^{\dagger}a_{r}^{\dagger}|psi>
    """
    rho, rdm_generator, transform, molecule = system()
    dim = molecule.n_qubits

    thdm_aa = np.zeros((int(dim/2), int(dim/2), int(dim/2), int(dim/2)),
                       dtype=complex)
    thdm_bb = np.zeros((int(dim/2), int(dim/2), int(dim/2), int(dim/2)),
                       dtype=complex)
    thdm_ab = np.zeros((int(dim/2), int(dim/2), int(dim/2), int(dim/2)),
                       dtype=complex)
    thdm_ba = np.zeros((int(dim/2), int(dim/2), int(dim/2), int(dim/2)),
                       dtype=complex)
    for p, q, r, s in product(range(int(dim/2)), repeat=4):
        # pauli_proj_op = transform.product_ops([2 * p + 1, 2 * q + 1,
        #                                        2 * s + 1, 2 * r + 1], [1, 1, -1, -1])
        pauli_proj_op = transform(FermionOperator(((2 * p + 1, 0), (2 * q + 1, 0), (2 * s + 1, 1), (2 * r + 1, 1))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        tpdm_element = np.trace(lifted_op.dot(rho))
        thdm_bb[p, q, r, s] = tpdm_element

        # pauli_proj_op = transform.product_ops([2 * p, 2 * q,
        #                                        2 * s, 2 * r], [1, 1, -1, -1])
        pauli_proj_op = transform(FermionOperator(((2 * p, 0), (2 * q, 0), (2 * s, 1), (2 * r, 1))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        tpdm_element = np.trace(lifted_op.dot(rho))
        thdm_aa[p, q, r, s] = tpdm_element

        # pauli_proj_op = transform.product_ops([2 * p, 2 * q + 1,
        #                                        2 * s + 1, 2 * r], [1, 1, -1, -1])
        pauli_proj_op = transform(FermionOperator(((2 * p, 0), (2 * q + 1, 0), (2 * s + 1, 1), (2 * r, 1))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        tpdm_element = np.trace(lifted_op.dot(rho))
        thdm_ab[p, q, r, s] = tpdm_element

        # pauli_proj_op = transform.product_ops([2 * p + 1, 2 * q,
        #                                        2 * s, 2 * r + 1], [1, 1, -1, -1])
        pauli_proj_op = transform(FermionOperator(((2 * p + 1, 0), (2 * q, 0), (2 * s, 1), (2 * r + 1, 1))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        tpdm_element = np.trace(lifted_op.dot(rho))
        thdm_ba[p, q, r, s] = tpdm_element

    thdm_aa_true, thdm_bb_true, thdm_ab_true, thdm_ba_true = rdm_generator.construct_thdm()
    assert np.allclose(thdm_aa, thdm_aa_true)
    assert np.allclose(thdm_bb, thdm_bb_true)
    assert np.allclose(thdm_ab, thdm_ab_true)
    assert np.allclose(thdm_ba, thdm_ba_true)


def test_construct_phdm():
    """
    Construct the particle-hole density matrix

    <psi|a_{p}^{\dagger}a_{q}a_{s}^{\dagger}a_{r}|psi>
    """
    rho, rdm_generator, transform, molecule = system()
    dim = molecule.n_qubits

    phdm_ab = np.zeros((int(dim/2), int(dim/2), int(dim/2), int(dim/2)), dtype=complex)
    phdm_ba = np.zeros((int(dim/2), int(dim/2), int(dim/2), int(dim/2)), dtype=complex)
    phdm_aabb = np.zeros((2 * (int(dim/2))**2, 2 * (int(dim/2))**2), dtype=complex)
    sdim = int(dim/2)
    for p, q, r, s in product(range(sdim), repeat=4):
        # pauli_proj_op = transform.product_ops([2 * p, 2 * q + 1,
        #                                        2 * s + 1, 2 * r], [-1, 1, -1, 1])
        pauli_proj_op = transform(FermionOperator(((2 * p, 1), (2 * q + 1, 0), (2 * s + 1, 1), (2 * r, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        phdm_element = np.trace(lifted_op.dot(rho))
        phdm_ab[p, q, r, s] = phdm_element

        # pauli_proj_op = transform.product_ops([2 * p + 1, 2 * q,
        #                                        2 * s, 2 * r + 1], [-1, 1, -1, 1])
        pauli_proj_op = transform(FermionOperator(((2 * p + 1, 1), (2 * q, 0), (2 * s, 1), (2 * r + 1, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        phdm_element = np.trace(lifted_op.dot(rho))
        phdm_ba[p, q, r, s] = phdm_element

        # pauli_proj_op = transform.product_ops([2 * p, 2 * q,
        #                                        2 * s, 2 * r], [-1, 1, -1, 1])
        pauli_proj_op = transform(FermionOperator(((2 * p, 1), (2 * q, 0), (2 * s, 1), (2 * r, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        phdm_element = np.trace(lifted_op.dot(rho))
        # phdm_aabb[0, 0, p, q, r, s] = phdm_element
        phdm_aabb[p * sdim + q, r * sdim + s] = phdm_element

        # pauli_proj_op = transform.product_ops([2 * p + 1, 2 * q + 1,
        #                                        2 * s + 1, 2 * r + 1], [-1, 1, -1, 1])
        pauli_proj_op = transform(FermionOperator(((2 * p + 1, 1), (2 * q + 1, 0), (2 * s + 1, 1), (2 * r + 1, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        phdm_element = np.trace(lifted_op.dot(rho))
        # phdm_aabb[1, 1, p, q, r, s] = phdm_element
        phdm_aabb[p * sdim + q + sdim**2, r * sdim + s + sdim**2] = phdm_element

        # pauli_proj_op = transform.product_ops([2 * p, 2 * q,
        #                                        2 * s + 1, 2 * r + 1], [-1, 1, -1, 1])
        pauli_proj_op = transform(FermionOperator(((2 * p, 1), (2 * q, 0), (2 * s + 1, 1), (2 * r + 1, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        phdm_element = np.trace(lifted_op.dot(rho))
        # phdm_aabb[0, 1, p, q, r, s] = phdm_element
        phdm_aabb[p * sdim + q, r * sdim + s + sdim**2] = phdm_element


        # pauli_proj_op = transform.product_ops([2 * p + 1, 2 * q + 1,
        #                                        2 * s, 2 * r], [-1, 1, -1, 1])
        pauli_proj_op = transform(FermionOperator(((2 * p + 1, 1), (2 * q + 1, 0), (2 * s, 1), (2 * r,  0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        phdm_element = np.trace(lifted_op.dot(rho))
        # phdm_aabb[1, 0, p, q, r, s] = phdm_element
        phdm_aabb[p * sdim + q + sdim**2, r * sdim + s] = phdm_element

    phdm_ab_test, phdm_ba_test, phdm_aabb_test = rdm_generator.construct_phdm()

    assert np.allclose(phdm_ab, phdm_ab_test)
    assert np.allclose(phdm_ba, phdm_ba_test)
    assert np.allclose(phdm_aabb, phdm_aabb_test)
