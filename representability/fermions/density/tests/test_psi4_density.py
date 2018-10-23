"""Test if psi4 density is returning a close result"""
import sys
import os
import numpy as np
from openfermion.hamiltonians import MolecularData
from openfermion.utils import map_two_pdm_to_two_hole_dm, map_two_pdm_to_particle_hole_dm
from representability.fermions.density.psi4_density import Psi4SpinOrbitalDensity
from representability.fermions.density.spin_density import SpinOrbitalDensity
from representability.fermions.utils import get_molecule_openfermion
from representability.config import DATA_DIRECTORY


def test_correct_marginals_h2():
    geom = [('H', (0., 0.0, 0.0)), ('H', (0., 0.0, 0.75))]
    basis = 'sto-3g'
    charge = 0
    multiplicity = 1
    of_molecule = MolecularData(geom, basis, multiplicity, charge)
    density = Psi4SpinOrbitalDensity(of_molecule)
    opdm = density.construct_opdm()
    oqdm = density.construct_ohdm()
    tpdm = density.construct_tpdm()
    tqdm = density.construct_thdm()
    phdm = density.construct_phdm()
    # this is supposed to be [[I, ^{2}D], [^{2}D, 0]]
    error = density.construct_tpdm_error_matrix(tpdm)

    h2_file = os.path.join(DATA_DIRECTORY, 'H2_sto-3g_singlet_0.75.hdf5')
    molecule = MolecularData(filename=h2_file)
    dim = molecule.n_qubits
    opdm_truth = molecule.fci_one_rdm
    oqdm_truth = np.eye(molecule.n_qubits) - opdm_truth
    tpdm_truth = np.einsum('ijkl->ijlk', molecule.fci_two_rdm)
    tqdm_truth = np.einsum('ijkl->ijlk', map_two_pdm_to_two_hole_dm(molecule.fci_two_rdm, molecule.fci_one_rdm))
    phdm_truth = np.einsum('ijkl->ijlk', map_two_pdm_to_particle_hole_dm(molecule.fci_two_rdm, molecule.fci_one_rdm))

    assert np.allclose(error[:dim**2, :dim**2], np.eye(dim**2))
    assert np.allclose(error[:dim**2, dim**2:], tpdm_truth.reshape((dim**2, dim**2)))
    assert np.allclose(error[dim**2:, :dim**2], tpdm_truth.reshape((dim**2, dim**2)).T)
    assert np.allclose(error[dim**2:, dim**2:], np.zeros((dim**2, dim**2)))
    np.testing.assert_allclose(opdm_truth.real, opdm.real, atol=1.0E-6)
    np.testing.assert_allclose(oqdm_truth.real, oqdm.real, atol=1.0E-6)
    np.testing.assert_allclose(tpdm_truth.real, tpdm.real, atol=1.0E-6)
    np.testing.assert_allclose(tqdm_truth.real, tqdm.real, atol=1.0E-6)
    np.testing.assert_allclose(phdm_truth.real, phdm.real, atol=1.0E-6)


def test_correct_marginals_heh():
    geom = [('H', (0., 0.0, 0.0)), ('He', (0., 0.0, 0.740848149))]
    basis = 'sto-3g'
    charge = 1
    multiplicity = 1
    calc_type = 'fci'
    of_molecule = MolecularData(geom, basis, multiplicity, charge)
    density = Psi4SpinOrbitalDensity(of_molecule)
    opdm = density.construct_opdm()
    oqdm = density.construct_ohdm()
    tpdm = density.construct_tpdm()
    tqdm = density.construct_thdm()
    phdm = density.construct_phdm()
    error = density.construct_tpdm_error_matrix(tpdm)


    heh_file = os.path.join(DATA_DIRECTORY, 'H1-He1_sto-3g_singlet_1+_0.74.hdf5')
    molecule = MolecularData(filename=heh_file)
    dim = molecule.n_qubits
    opdm_truth = molecule.fci_one_rdm
    oqdm_truth = np.eye(molecule.n_qubits) - opdm_truth
    tpdm_truth = np.einsum('ijkl->ijlk', molecule.fci_two_rdm)
    tqdm_truth = np.einsum('ijkl->ijlk', map_two_pdm_to_two_hole_dm(molecule.fci_two_rdm, molecule.fci_one_rdm))
    phdm_truth = np.einsum('ijkl->ijlk', map_two_pdm_to_particle_hole_dm(molecule.fci_two_rdm, molecule.fci_one_rdm))

    assert np.allclose(error[:dim**2, :dim**2], np.eye(dim**2))
    assert np.allclose(error[:dim**2, dim**2:], tpdm_truth.reshape((dim**2, dim**2)))
    assert np.allclose(error[dim**2:, :dim**2], tpdm_truth.reshape((dim**2, dim**2)).T)
    assert np.allclose(error[dim**2:, dim**2:], np.zeros((dim**2, dim**2)))
    np.testing.assert_allclose(opdm_truth.real, opdm.real, atol=1.0E-11)
    np.testing.assert_allclose(oqdm_truth.real, oqdm.real, atol=1.0E-11)
    np.testing.assert_allclose(tpdm_truth.real, tpdm.real, atol=1.0E-11)
    np.testing.assert_allclose(tqdm_truth.real, tqdm.real, atol=1.0E-11)
    np.testing.assert_allclose(phdm_truth.real, phdm.real, atol=1.0E-11)

