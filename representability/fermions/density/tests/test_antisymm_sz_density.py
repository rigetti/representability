"""
Testing the construction of tpdm_aa, tpdm_bb on antisymmetric support

For the antisymmetric symmetry adapting we only need to test these two functions
because all other functions are inherited from `SymmOrbitalDensity` which has a sepearte
and comprehensive testing suite.
"""
import sys
from itertools import product
import numpy as np
from grove.alpha.fermion_transforms.jwtransform import JWTransform
from referenceqvm.unitary_generator import tensor_up
from pyquil.paulis import PauliTerm, PauliSum

from representability.fermions.density.antisymm_sz_density import AntiSymmOrbitalDensity
from representability.fermions.density.spin_density import SpinOrbitalDensity
from representability.config import *
from representability.fermions.utils import get_molecule_openfermion
from representability.fermions.density.antisymm_sz_density import unspin_adapt
from representability.fermions.density.antisymm_sz_maps import get_sz_spin_adapted

from openfermion.transforms import jordan_wigner
from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator
from openfermionpsi4 import run_psi4

from forestopenfermion.pyquil_connector import qubitop_to_pyquilpauli


# probably want to upgrade this with yield fixture.  This will need to be an object
def system():
    print('Running System Setup')
    basis = 'sto-3g'
    multiplicity = 1
    charge = 1
    geometry = [('H', [0.0, 0.0, 0.0]), ('He', [0, 0, 0.740848149])]
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
    rdm_generator = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    transform = jordan_wigner
    return n_density, rdm_generator, transform, molecule


def test_construct_tpdm():
    """
    Construct the two-particle density matrix

    <psi|a_{p}^{\dagger}a_{q}^{\dagger}a_{s}a_{r}|psi>
    """
    # aggregate the system
    rho, rdm_generator, transform, molecule = system()
    dim = molecule.n_qubits
    # get the spin-orbital density matrix 
    spin_orbital_dim = int(dim)
    tpdm = np.zeros((spin_orbital_dim, spin_orbital_dim, spin_orbital_dim, spin_orbital_dim))
    for i, j, k, l in product(range(spin_orbital_dim), repeat=4):
        pauli_proj_op = transform(FermionOperator(((i, 1), (j, 1), (l, 0), (k, 0))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        tpdm_element = np.trace(lifted_op.dot(rho))
        assert np.isclose(tpdm_element.imag, 0.0)
        tpdm[i, j, k, l] = tpdm_element.real

    # put into spin-adapted RDMS
    # index is spatial orbital index
    m_dim = int(dim/2)
    d2_aa_true = np.zeros((int(m_dim * (m_dim - 1)/2), int(m_dim * (m_dim - 1)/2)))
    d2_bb_true = np.zeros((int(m_dim * (m_dim - 1)/2), int(m_dim * (m_dim - 1)/2)))
    d2_ab_true = np.zeros((int(m_dim**2), int(m_dim**2)))

    # build basis look up table
    bas_aa = {}
    bas_ab = {}
    cnt_aa = 0
    cnt_ab = 0
    for p, q in product(range(m_dim), repeat=2):
        if q > p:
            bas_aa[(p, q)] = cnt_aa
            cnt_aa += 1
        bas_ab[(p, q)] = cnt_ab
        cnt_ab += 1

    for i, j, k, l in product(range(m_dim), repeat=4):
        d2_ab_true[bas_ab[(i, j)], bas_ab[(k, l)]] = tpdm[2 * i, 2 * j + 1, 2 * k, 2 * l + 1]
        if i < j and k < l:
            d2_aa_true[bas_aa[(i, j)], bas_aa[(k, l)]] = tpdm[2 * i, 2 * j, 2 * k, 2 * l] - \
                                                         tpdm[2 * i, 2 * j, 2 * l, 2 * k] - \
                                                         tpdm[2 * j, 2 * i, 2 * k, 2 * l] + \
                                                         tpdm[2 * j, 2 * i, 2 * l, 2 * k]

            assert np.allclose([-tpdm[2 * i, 2 * j, 2 * l, 2 * k],
                                -tpdm[2 * j, 2 * i, 2 * k, 2 * l],
                                 tpdm[2 * j, 2 * i, 2 * l, 2 * k]],
                                 tpdm[2 * i, 2 * j, 2 * k, 2 * l])

            d2_bb_true[bas_aa[(i, j)], bas_aa[(k, l)]] = tpdm[2*i+1, 2*j+1, 2*k+1, 2*l+1] - \
                                                         tpdm[2*i+1, 2*j+1, 2*l+1, 2*k+1] - \
                                                         tpdm[2*j+1, 2*i+1, 2*k+1, 2*l+1] + \
                                                         tpdm[2*j+1, 2*i+1, 2*l+1, 2*k+1]

            d2_aa_true[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5
            d2_bb_true[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5


    # get the output from the rdm_generator
    tpdm_aa, tpdm_bb, tpdm_ab, [bas_aa, bas_ab] = rdm_generator.construct_tpdm()
    assert np.allclose(tpdm_aa, d2_aa_true)
    assert np.allclose(tpdm_bb, d2_bb_true)
    assert np.allclose(tpdm_ab, d2_ab_true)


def test_construct_thdm():
    """
    Construct the two-hole density matrix

    <psi|a_{p}a_{q}a_{s}^{\dagger}a_{r}^{\dagger}|psi>
    """
    # aggregate the system
    rho, rdm_generator, transform, molecule = system()
    dim = molecule.n_qubits
    # get the spin-orbital density matrix
    spin_orbital_dim = int(dim)
    jw = JWTransform()
    tqdm = np.zeros((spin_orbital_dim, spin_orbital_dim, spin_orbital_dim, spin_orbital_dim))
    for i, j, k, l in product(range(spin_orbital_dim), repeat=4):
        # pauli_proj_op = jw.product_ops([i, j, l, k], [1, 1, -1, -1])
        pauli_proj_op = transform(FermionOperator(((i, 0), (j, 0), (l, 1), (k, 1))))
        pauli_proj_op = qubitop_to_pyquilpauli(pauli_proj_op)
        if isinstance(pauli_proj_op, PauliTerm):
            pauli_proj_op = PauliSum([pauli_proj_op])
        lifted_op = tensor_up(pauli_proj_op, molecule.n_qubits)
        tqdm_element = np.trace(lifted_op.dot(rho))
        assert np.isclose(tqdm_element.imag, 0.0)
        tqdm[i, j, k, l] = tqdm_element.real

    # put into spin-adapted RDMS
    # index is spatial orbital index
    m_dim = int(dim/2)
    aa_dim = int(m_dim * (m_dim - 1)/2)
    ab_dim = int(m_dim**2)
    q2_aa_true = np.zeros((aa_dim, aa_dim))
    q2_bb_true = np.zeros((aa_dim, aa_dim))
    q2_ab_true = np.zeros((ab_dim, ab_dim))

    # build basis look up table
    bas_aa = {}
    bas_ab = {}
    cnt_aa = 0
    cnt_ab = 0
    for p, q in product(range(m_dim), repeat=2):
        if q > p:
            bas_aa[(p, q)] = cnt_aa
            cnt_aa += 1
        bas_ab[(p, q)] = cnt_ab
        cnt_ab += 1

    for i, j, k, l in product(range(m_dim), repeat=4):
        q2_ab_true[bas_ab[(i, j)], bas_ab[(k, l)]] = tqdm[2 * i, 2 * j + 1, 2 * k, 2 * l + 1]
        if i < j and k < l:
            q2_aa_true[bas_aa[(i, j)], bas_aa[(k, l)]] = tqdm[2 * i, 2 * j, 2 * k, 2 * l] - \
                                                         tqdm[2 * i, 2 * j, 2 * l, 2 * k] - \
                                                         tqdm[2 * j, 2 * i, 2 * k, 2 * l] + \
                                                         tqdm[2 * j, 2 * i, 2 * l, 2 * k]

            assert np.allclose([-tqdm[2 * i, 2 * j, 2 * l, 2 * k],
                                -tqdm[2 * j, 2 * i, 2 * k, 2 * l],
                                 tqdm[2 * j, 2 * i, 2 * l, 2 * k]],
                                 tqdm[2 * i, 2 * j, 2 * k, 2 * l])

            q2_bb_true[bas_aa[(i, j)], bas_aa[(k, l)]] = tqdm[2*i+1, 2*j+1, 2*k+1, 2*l+1] - \
                                                         tqdm[2*i+1, 2*j+1, 2*l+1, 2*k+1] - \
                                                         tqdm[2*j+1, 2*i+1, 2*k+1, 2*l+1] + \
                                                         tqdm[2*j+1, 2*i+1, 2*l+1, 2*k+1]

            q2_aa_true[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5
            q2_bb_true[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5


    # get the output from the rdm_generator
    tqdm_aa, tqdm_bb, tqdm_ab, [bas_aa, bas_ab] = rdm_generator.construct_thdm()
    assert np.allclose(tqdm_aa, q2_aa_true)
    assert np.allclose(tqdm_bb, q2_bb_true)
    assert np.allclose(tqdm_ab, q2_ab_true)


def test_unspin_adapt_1():
    heh_file = os.path.join(DATA_DIRECTORY, 'H1-He1_sto-3g_singlet_1+_0.74.hdf5')
    molecule = MolecularData(filename=heh_file)
    tpdm_from_of = np.einsum('ijkl->ijlk', molecule.fci_two_rdm)
    rho, rdm_generator, transform, mol2 = system()
    assert np.allclose(np.conj(rho).T, rho)

    dim = molecule.n_qubits
    rdm_generator_antisym = AntiSymmOrbitalDensity(rho, molecule.n_qubits)
    rdm_generator_spin_orbital = SpinOrbitalDensity(rho, molecule.n_qubits)
    tpdm_aa, tpdm_bb, tpdm_ab, [bas_aa, bas_ab] = rdm_generator_antisym.construct_tpdm()
    tpdm = rdm_generator_spin_orbital.construct_tpdm()
    opdm = rdm_generator_spin_orbital.construct_opdm()

    assert np.allclose(opdm, molecule.fci_one_rdm)
    assert np.allclose(tpdm, tpdm_from_of)

    tpdm_test = unspin_adapt(tpdm_aa, tpdm_bb, tpdm_ab)
    # print(np.linalg.norm(tpdm[::2, ::2, ::2, ::2] - tpdm_test[::2, ::2, ::2, ::2]))
    np.testing.assert_allclose(tpdm[::2, ::2, ::2, ::2], tpdm_test[::2, ::2, ::2, ::2])
    # print(np.linalg.norm(tpdm[1::2, 1::2, 1::2, 1::2] - tpdm_test[1::2, 1::2, 1::2, 1::2]))
    np.testing.assert_allclose(tpdm[1::2, 1::2, 1::2, 1::2], tpdm_test[1::2, 1::2, 1::2, 1::2])
    # print(np.linalg.norm(tpdm[::2, 1::2, ::2, 1::2] - tpdm_test[::2, 1::2, ::2, 1::2]))
    np.testing.assert_allclose(tpdm[::2, 1::2, ::2, 1::2], tpdm_test[::2, 1::2, ::2, 1::2])
    # print(np.linalg.norm(tpdm[1::2, ::2, 1::2, ::2] - tpdm_test[1::2, ::2, 1::2, ::2]))
    np.testing.assert_allclose(tpdm[1::2, ::2, 1::2, ::2], tpdm_test[1::2, ::2, 1::2, ::2])
    for p, q, r, s in product(range(tpdm.shape[0]), repeat=4):
        if not np.isclose(tpdm[p, q, r, s], tpdm_test[p, q, r, s]):
            print((p, q, r, s), tpdm[p, q, r, s], tpdm_test[p, q, r, s])

    heh_file = os.path.join(DATA_DIRECTORY, 'H1-He1_sto-3g_singlet_1+_0.74.hdf5')
    molecule = MolecularData(filename=heh_file)
    tpdm_from_of = np.einsum('ijkl->ijlk', molecule.fci_two_rdm)
    for p, q, r, s in product(range(tpdm.shape[0]), repeat=4):
        if not np.isclose(tpdm_from_of[p, q, r, s], tpdm[p, q, r, s]):
            print((p, q, r, s), tpdm_from_of[p, q, r, s], tpdm[p, q, r, s])

    t_tpdm_aa, t_tpdm_bb, t_tpdm_ab = get_sz_spin_adapted(tpdm)
    assert np.allclose(t_tpdm_ab, tpdm_ab)
