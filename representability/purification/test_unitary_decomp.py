"""Test the accuracy and exceptions of unitary_decomp.py"""
import numpy as np
import pytest
from itertools import product
from representability.purification.unitary_decomp import (coleman_decomposition,
                                                    mazziotti_opdm_purification,
                                                    decrease_homo_to_target,
                                                    increase_lumo_to_target)

from representability.purification.unitary_subspace_purification import (
    unitary_subspace_purification_fixed_initial_trace, coleman_projection_dq)
from representability.fermions.density.spin_density import SpinOrbitalDensity
from representability.fermions.utils import (get_molecule_openfermion,
                                             check_antisymmetric_d2,
                                             wedge_product, four_tensor2matrix)
from representability.sampling import (add_gaussian_noise,
                                add_gaussian_noise_antisymmetric_four_tensor)
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner
from openfermionpsi4 import run_psi4


def heh_system():
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
    rdm_generator = SpinOrbitalDensity(n_density, molecule.n_qubits)
    transform = jordan_wigner
    return n_density, rdm_generator, transform, molecule


def test_coleman_decomp():
    n_density, rdm_gen, transform, molecule = heh_system()
    density = SpinOrbitalDensity(n_density, molecule.n_qubits)
    tpdm = density.construct_tpdm()
    opdm = density.construct_opdm()
    dim = tpdm.shape[0]

    # check if we get a valid tpdm
    check_antisymmetric_d2(tpdm)

    # now test the Coleman-Absar decomposition
    d2_0, d2_1, d2_2 = coleman_decomposition(tpdm)

    # now check some properties.
    # 1. check d2_2 contracts to zero
    d1_zero = np.einsum('ikjk', d2_2)
    assert np.allclose(d1_zero, 0.0)

    # 2. check orthogonality of tenors
    assert np.isclose(np.einsum('ijkl, lkji', d2_0, d2_1), 0.0)
    assert np.isclose(np.einsum('ijkl, lkji', d2_1, d2_2), 0.0)
    assert np.isclose(np.einsum('ijkl, lkji', d2_2, d2_0), 0.0)

    # 3. check reconstruct to tpdm
    assert np.allclose(d2_0 + d2_1 + d2_2, tpdm)

    # 4. check D1 looks like a contracted d2 up to coefficients
    one_one_tensor = np.einsum('ikjk', tpdm)
    N = molecule.n_electrons
    # why are we missing the factor of 2? Is it because of normalization on D2?
    assert np.allclose(one_one_tensor, (N - 1) * opdm)

    # 5. Check trace of d2_0 is same as tpdm
    assert np.isclose(np.einsum('ijij', tpdm), np.einsum('ijij', d2_0))

    # 6. Check one-trace (d2_0 + d2_1) = a1
    assert np.allclose(np.einsum('ikjk', d2_0 + d2_1), one_one_tensor)


def test_mazziotti_opdm_purification():
    n_density, rdm_gen, transform, molecule = heh_system()
    density = SpinOrbitalDensity(n_density, molecule.n_qubits)
    opdm = density.construct_opdm()
    opdm_corrupted = add_gaussian_noise(opdm, 0.001)
    opdm_corrupted = 0.5 * (opdm_corrupted + np.conj(opdm_corrupted).T)
    purificed_opdm = mazziotti_opdm_purification(opdm_corrupted, opdm.trace().real)
    assert np.isclose(purificed_opdm.trace(), opdm.trace())
    w, v = np.linalg.eigh(purificed_opdm)
    assert (1 >= w).all()
    assert (w >= 0).all()


def test_decrease_homo_to_target():
    """test the make occs representable code"""
    eigvals = np.array([1.1, 0.8, 0.2, 0.1])
    eigvals = decrease_homo_to_target(eigvals, 2.0)
    assert np.isclose(np.sum(eigvals), 2.0)


def test_increase_lumo():
    eigvals = np.array([0., 0., 0., 0.8,  1., 1.])
    eigvals = increase_lumo_to_target(eigvals, 3)
    assert np.isclose(np.sum(eigvals), 3.0)


@pytest.mark.skip(reason="System Test")
def test_unitary_subspace_purification_fixed_opdm():
    n_density, rdm_gen, transform, molecule = heh_system()
    density = SpinOrbitalDensity(n_density, molecule.n_qubits)
    tpdm = density.construct_tpdm()
    tpdm_noise = add_gaussian_noise_antisymmetric_four_tensor(tpdm, 0.00001)
    check_antisymmetric_d2(tpdm_noise.real)
    tpdm_projected = unitary_subspace_purification_fixed_initial_trace(
        tpdm_noise.real, molecule.n_electrons, disp=True)

    d2_matrix = four_tensor2matrix(tpdm_projected)
    w, v = np.linalg.eigh(d2_matrix)
    assert all(w > -1.0E-6)
    # np.testing.assert_allclose(tpdm_projected, tpdm_noise, atol=1.0E-4)


def test_unitary_subspace_purification_dq():
    n_density, rdm_gen, transform, molecule = heh_system()
    density = SpinOrbitalDensity(n_density, molecule.n_qubits)
    tpdm = density.construct_tpdm()
    tpdm_noise = add_gaussian_noise_antisymmetric_four_tensor(tpdm, 0.00001)
    check_antisymmetric_d2(tpdm_noise.real)
    tpdm_projected = coleman_projection_dq(
        tpdm_noise.real, molecule.n_electrons, disp=True)

    d2_matrix = four_tensor2matrix(tpdm_projected)
    w, v = np.linalg.eigh(d2_matrix)
    assert all(w > -1.0E-6)


def test_map_d2_q2_with_wedge():
    n_density, rdm_gen, transform, molecule = heh_system()
    density = SpinOrbitalDensity(n_density, molecule.n_qubits)
    tpdm = density.construct_tpdm()
    tqdm = density.construct_thdm()
    opdm = density.construct_opdm()
    dim = opdm.shape[0]
    eye_wedge_eye = wedge_product(np.eye(dim), np.eye(dim))
    one_wedge_eye = wedge_product(opdm, np.eye(dim))
    tqdm_test = 2 * eye_wedge_eye - 4 * one_wedge_eye + tpdm
    for p, q, r, s in product(range(dim), repeat=4):
        np.testing.assert_allclose(tqdm_test[p, q, r, s], tqdm[p, q, r, s], atol=1.0E-10)
