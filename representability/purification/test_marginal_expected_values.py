import numpy as np
from representability.fermions.density.spin_density import SpinOrbitalDensity
from representability.fermions.utils import get_molecule_openfermion
from representability.purification.marginal_expected_values import (sz_expected,
                    number_expectation, s2_expected)
from grove.alpha.fermion_transforms.jwtransform import JWTransform
from pyquil.paulis import sI
from referenceqvm.unitary_generator import tensor_up
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


def test_sz_expected():
    n_density, rdm_gen, transform, molecule = heh_system()

    density = SpinOrbitalDensity(n_density, molecule.n_qubits)
    tpdm = density.construct_tpdm().real
    opdm = density.construct_opdm().real

    # calculate on the fly
    sz_eval = 0.0
    for i in range(opdm.shape[0]//2):
        sz_eval += opdm[2 * i, 2 * i] - opdm[2 * i + 1, 2 * i + 1]
    sz_eval *= 0.5

    np.testing.assert_almost_equal(sz_expected(opdm), sz_eval)


def test_number_expected():
    n_density, rdm_gen, transform, molecule = heh_system()

    density = SpinOrbitalDensity(n_density, molecule.n_qubits)
    tpdm = density.construct_tpdm().real
    opdm = density.construct_opdm().real

    # calculate on the fly
    np.testing.assert_almost_equal(number_expectation(opdm), opdm.trace())


def test_s2_expected():
    n_density, rdm_gen, transform, molecule = heh_system()

    density = SpinOrbitalDensity(n_density, molecule.n_qubits)
    tpdm = density.construct_tpdm().real
    opdm = density.construct_opdm().real

    # calculate S^{2} by jordan-wigner of fermionic modes
    so_dim = int(opdm.shape[0])
    sp_dim = int(so_dim / 2)
    jw = JWTransform()
    sminus = sI(0) * 0
    splus = sI(0) * 0
    szop = sI(0) * 0
    for i in range(sp_dim):
        sminus += jw.product_ops([2 * i + 1, 2 * i], [-1, 1])
        splus += jw.product_ops([2 * i, 2 * i + 1], [-1, 1])
        szop += 0.5 * jw.product_ops([2 * i, 2 * i], [-1, 1])
        szop -= 0.5 * jw.product_ops([2 * i + 1, 2 * i + 1], [-1, 1])

    s2op = sminus * splus + szop + szop * szop
    s2op_matrix = tensor_up(s2op, so_dim)
    szop_matrix = tensor_up(szop, so_dim)
    szop2_matrix = tensor_up(szop * szop, so_dim)
    splus_matrix = tensor_up(splus, so_dim)
    sminus_matrix = tensor_up(sminus, so_dim)
    smsp_matrix = tensor_up(sminus * splus, so_dim)

    np.testing.assert_allclose(smsp_matrix, sminus_matrix.dot(splus_matrix))
    np.testing.assert_allclose(szop2_matrix, szop_matrix.dot(szop_matrix))
    np.testing.assert_allclose(szop2_matrix + szop_matrix,
                               szop_matrix.dot(szop_matrix) + szop_matrix)

    np.testing.assert_allclose(smsp_matrix + szop2_matrix + szop_matrix,
                               smsp_matrix + szop_matrix.dot(szop_matrix) + szop_matrix)

    np.testing.assert_allclose(smsp_matrix + szop2_matrix + szop_matrix,
                               s2op_matrix)

    # Now we can check if our two methods for computing the S^{2} value are
    # correct.
    s2_from_ndensity = np.trace(n_density.dot(s2op_matrix)).real
    trial_s2_expected = s2_expected(tpdm, opdm)
    np.testing.assert_almost_equal(s2_from_ndensity, trial_s2_expected)

