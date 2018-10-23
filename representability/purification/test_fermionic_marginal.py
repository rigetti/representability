"""
Test the purification of a Fermionic marginal
"""
from itertools import product
import pytest
import numpy as np
from representability.fermions.utils import get_molecule_openfermion
from representability.fermions.density.spin_density import SpinOrbitalDensity
from representability.sampling import add_gaussian_noise_antisymmetric_four_tensor
from representability.purification.fermionic_marginal import purify_marginal, map_tpdm_to_opdm, map_tpdm_to_tqdm, \
                                                             map_tpdm_to_phdm, map_tqdm_to_tpdm, map_phdm_to_tpdm, \
                                                             symmeterize_matrix, symmeterize_four_tensor, \
                                                             antisymmeterizer, map_tqdm_to_oqdm, map_oqdm_to_opdm, \
                                                             map_opdm_to_oqdm

from representability.fermions.utils import check_antisymmetric_d2

from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner
from openfermionpsi4 import run_psi4


def h2_system():
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
    rdm_generator = SpinOrbitalDensity(n_density, molecule.n_qubits)
    transform = jordan_wigner
    return n_density, rdm_generator, transform, molecule


def test_symmeterize_matrix():
    dim = 10
    np.random.seed(42)
    mat = np.random.random((dim**2, dim**2))
    sym_mat = symmeterize_matrix(mat)
    assert np.allclose(sym_mat, 0.5 * (mat + mat.T))


def test_symmeterize_tensor():
    dim = 10
    np.random.seed(42)
    mat = np.random.random((dim, dim, dim, dim))
    sym_mat = symmeterize_four_tensor(mat)
    true_sym = np.zeros_like(mat)
    for p, q, r, s in product(range(dim), repeat=4):
        true_sym[p, q, r, s] = 0.5 * (mat[p, q, r, s] + mat[r, s, p, q])
        true_sym[r, s, p, q] = true_sym[p, q, r, s]
    assert np.allclose(sym_mat, true_sym)


def test_antisymmeterizer():
    dim = 10
    np.random.seed(42)
    mat = np.random.random((dim, dim, dim, dim))
    sym_mat = symmeterize_four_tensor(mat)
    mat_anti = antisymmeterizer(sym_mat)
    check_antisymmetric_d2(mat_anti)


def test_purification_identity():
    n_density, rdm_gen, transform, molecule = h2_system()

    density = SpinOrbitalDensity(n_density, molecule.n_qubits)
    tpdm = density.construct_tpdm().real

    test_tpdm = purify_marginal(tpdm, molecule.n_electrons, molecule.n_qubits)
    assert np.allclose(test_tpdm, tpdm)


@pytest.mark.skip(reason="System Test")
def test_purification():
    n_density, rdm_gen, transform, molecule = h2_system()

    density = SpinOrbitalDensity(n_density, molecule.n_qubits)
    tpdm = density.construct_tpdm().real

    corrupted_tpdm = add_gaussian_noise_antisymmetric_four_tensor(tpdm.real, 0.0001)
    test_tpdm = purify_marginal(corrupted_tpdm, molecule.n_electrons, molecule.n_qubits)
    assert np.allclose(test_tpdm, tpdm)


def test_mappings():
    n_density, rdm_gen, transform, molecule = h2_system()

    density = SpinOrbitalDensity(n_density, molecule.n_qubits)
    tpdm = density.construct_tpdm().real
    opdm = density.construct_opdm().real
    oqdm = density.construct_ohdm().real
    tqdm = density.construct_thdm().real
    phdm = density.construct_phdm().real

    test_opdm = map_tpdm_to_opdm(tpdm, molecule.n_electrons)
    assert np.allclose(test_opdm, opdm)

    test_tqdm = map_tpdm_to_tqdm(tpdm, opdm)
    assert np.allclose(test_tqdm, tqdm)

    test_phdm = map_tpdm_to_phdm(tpdm, opdm)
    assert np.allclose(test_phdm, phdm)

    test_tpdm = map_tqdm_to_tpdm(tqdm, opdm)
    assert np.allclose(test_tpdm, tpdm)

    test_tpdm = map_phdm_to_tpdm(phdm, opdm)
    assert np.allclose(test_tpdm, tpdm)

    test_oqdm = map_tqdm_to_oqdm(tqdm, molecule.n_qubits - molecule.n_electrons)
    assert np.allclose(test_oqdm, oqdm)

    test_opdm = map_oqdm_to_opdm(oqdm)
    assert np.allclose(test_opdm, opdm)

    test_oqdm = map_opdm_to_oqdm(opdm)
    assert np.allclose(test_oqdm, oqdm)

    eta = molecule.n_qubits - molecule.n_electrons
    N = molecule.n_electrons
    opdm_test = np.einsum('ijkj', phdm)/((N + 1))
    assert np.allclose(opdm, opdm_test)

