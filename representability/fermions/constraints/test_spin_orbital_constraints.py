import sys
import pytest
import numpy as np
import os
from scipy.sparse import csc_matrix
from itertools import product
from representability.fermions.constraints.spin_orbital_constraints import (
    d1_q1_mapping, d2_d1_mapping, d2_q2_mapping, d2_g2_mapping,
    antisymmetry_constraints, spin_orbital_linear_constraints,
    d2_e2_mapping, d2_to_t1, d2_to_t1_matrix, d2_to_t1_from_iterator, d2_to_t1_matrix_antisym)
from representability.tensor import Tensor
from representability.multitensor import MultiTensor
from representability.fermions.density.spin_density import SpinOrbitalDensity
from representability.fermions.density.spin_maps import map_d2_g2
from representability.fermions.utils import get_molecule_openfermion
from representability.fermions.basis_utils import (geminal_spin_basis,
                                                   antisymmetry_adapting,
                                                   triples_spin_orbital_antisymm_basis)
from representability.config import RDM_DIRECTORY, DATA_DIRECTORY
from representability.tensor import index_tuple_basis

from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner
from openfermion.utils import map_two_pdm_to_two_hole_dm, map_two_pdm_to_particle_hole_dm
from openfermionpsi4 import run_psi4


def test_d1_q1_mapping():
    dim = 2
    opdm = np.random.random((dim, dim))
    opdm = (opdm.T + opdm)/2
    oqdm = np.eye(dim) - opdm
    opdm = Tensor(opdm, name='ck')
    oqdm = Tensor(oqdm, name='kc')
    rdm = MultiTensor([opdm, oqdm])

    assert set([tt.name for tt in rdm.tensors]) == {'ck', 'kc'}
    assert np.allclose(rdm.tensors['ck'].data + rdm.tensors['kc'].data, np.eye(dim))

    assert np.isclose(rdm.vec_dim, 8)

    # get the dual basis mapping between these
    db = d1_q1_mapping(dim)

    rdm.dual_basis = db
    A, b, c = rdm.synthesize_dual_basis()
    Amat =  A.todense()
    bmat = b.todense()
    cmat = c.todense()

    primal_vec = rdm.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_d1():
    heh_file = os.path.join(DATA_DIRECTORY, 'H1-He1_sto-3g_singlet_1+_0.74.hdf5')
    molecule = MolecularData(filename=heh_file)
    tpdm = np.einsum('ijkl->ijlk', molecule.fci_two_rdm)
    opdm = molecule.fci_one_rdm

    tpdm = Tensor(tpdm, name='cckk')
    opdm = Tensor(opdm, name='ck')
    mt = MultiTensor([tpdm, opdm])

    db = d2_d1_mapping(molecule.n_qubits, (2 - 1))
    mt.dual_basis = db

    A, b, c = mt.synthesize_dual_basis()
    Amat = A.todense()
    bmat = b.todense()
    cmat = c.todense()

    primal_vec = mt.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    # now try doing it on a matrix tpdm with a given basis
    bb_aa, bb_ab = geminal_spin_basis(molecule.n_qubits)
    dim = molecule.n_qubits
    d2 = np.zeros((dim**2, dim**2))
    for p, q, r, s in product(range(dim), repeat=4):
        d2[bb_ab.rev((p, q)), bb_ab.rev((r, s))] = tpdm.data[p, q, r, s].real

    d2 = Tensor(d2, basis=bb_ab, name='cckk')

    rdms = MultiTensor([opdm, d2])
    rdms.dual_basis = db
    A, _, c = rdms.synthesize_dual_basis()
    Amat = A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_antisymm():
    heh_file = os.path.join(DATA_DIRECTORY, 'H1-He1_sto-3g_singlet_1+_0.74.hdf5')
    molecule = MolecularData(filename=heh_file)
    tpdm = np.einsum('ijkl->ijlk', molecule.fci_two_rdm)

    tpdm = Tensor(tpdm, name='cckk')
    mt = MultiTensor([tpdm])

    db = antisymmetry_constraints(molecule.n_qubits)
    mt.dual_basis = db

    A, b, c = mt.synthesize_dual_basis()
    Amat = A.todense()
    cmat = c.todense()

    primal_vec = mt.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_q2():
    heh_file = os.path.join(DATA_DIRECTORY, 'H1-He1_sto-3g_singlet_1+_0.74.hdf5')
    molecule = MolecularData(filename=heh_file)

    opdm = molecule.fci_one_rdm
    tpdm = np.einsum('ijkl->ijlk', molecule.fci_two_rdm)
    tqdm = np.einsum('ijkl->ijlk',
                     map_two_pdm_to_two_hole_dm(molecule.fci_two_rdm, molecule.fci_one_rdm))

    opdm = Tensor(opdm, name='ck')
    tpdm = Tensor(tpdm, name='cckk')
    tqdm = Tensor(tqdm, name='kkcc')

    rdms = MultiTensor([opdm, tpdm, tqdm])

    vec = np.vstack((opdm.data.reshape((-1, 1), order='C'), tpdm.data.reshape((-1, 1), order='C')))
    vec = np.vstack((vec, tqdm.data.reshape((-1, 1), order='C')))

    assert np.allclose(vec, rdms.vectorize_tensors())

    db = d2_q2_mapping(molecule.n_qubits)
    rdms.dual_basis = db
    A, _, c = rdms.synthesize_dual_basis()
    Amat = A.todense()
    cmat = c.todense()

    residual = Amat.dot(rdms.vectorize_tensors()) - cmat

    assert np.allclose(residual, np.zeros_like(residual))

    # now try doing it on a matrix tpdm with a given basis
    bb_aa, bb_ab = geminal_spin_basis(molecule.n_qubits)
    dim = molecule.n_qubits
    d2 = np.zeros((dim**2, dim**2))
    q2 = np.zeros((dim**2, dim**2))
    for p, q, r, s in product(range(dim), repeat=4):
        d2[bb_ab.rev((p, q)), bb_ab.rev((r, s))] = tpdm.data[p, q, r, s].real
        q2[bb_ab.rev((p, q)), bb_ab.rev((r, s))] = tqdm.data[p, q, r, s].real

    d2 = Tensor(d2, basis=bb_ab, name='cckk')
    q2 = Tensor(q2, basis=bb_ab, name='kkcc')

    rdms = MultiTensor([opdm, d2, q2])
    rdms.dual_basis = db
    A, b, c = rdms.synthesize_dual_basis()
    Amat = A.todense()
    bmat = b.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_g2():
    heh_file = os.path.join(DATA_DIRECTORY, 'H1-He1_sto-3g_singlet_1+_0.74.hdf5')
    molecule = MolecularData(filename=heh_file)

    opdm = molecule.fci_one_rdm
    tpdm = np.einsum('ijkl->ijlk', molecule.fci_two_rdm)
    phdm = np.einsum('ijkl->ijlk',
                     map_two_pdm_to_particle_hole_dm(molecule.fci_two_rdm, molecule.fci_one_rdm))

    tpdm = Tensor(tpdm, name='cckk')
    opdm = Tensor(opdm, name='ck')
    phdm = Tensor(phdm, name='ckck')

    test_vec_phdm = phdm.data.reshape((-1, 1), order='C')
    test_vec_phdm_2 = phdm.vectorize()
    assert np.allclose(test_vec_phdm, test_vec_phdm_2)

    phdm_test = map_d2_g2(tpdm.data, opdm.data)
    assert np.allclose(phdm_test, phdm.data)

    rdms = MultiTensor([opdm, tpdm, phdm])

    db = d2_g2_mapping(molecule.n_qubits)
    rdms.dual_basis = db
    A, _, c = rdms.synthesize_dual_basis()
    Amat = A.todense()
    cmat = c.todense()

    residual = Amat.dot(rdms.vectorize_tensors()) - cmat
    vec = rdms.vectorize_tensors()
    for i in range(Amat.shape[0]):
        if not np.isclose(Amat[i, :].dot(vec), 0):
            print(Amat[i, :].dot(vec), vars(db[i]))

    assert np.isclose(np.linalg.norm(residual), 0.0)
    assert np.allclose(residual, np.zeros_like(residual))

    # now try doing it on a matrix tpdm with a given basis
    dim = molecule.n_qubits
    bb_aa, bb_ab = geminal_spin_basis(dim)
    g2 = np.zeros((dim**2, dim**2))
    for p, q, r, s in product(range(dim), repeat=4):
        g2[bb_ab.rev((p, q)), bb_ab.rev((r, s))] = phdm.data[p, q, r, s].real

    g2 = Tensor(g2, basis=bb_ab, name='ckck')
    rdms = MultiTensor([opdm, tpdm, g2])
    rdms.dual_basis = db
    A, b, c = rdms.synthesize_dual_basis()
    Amat = A.todense()
    bmat = b.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


@pytest.mark.skip(reason="A Very expensive test")
def test_t1_construction():
    """
    Test if we map to the T1 matrix properly
    """
    phdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_phdm.npy'))
    dim = phdm.shape[0]
    # these are openfermion ordering.
    pphhdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_pphhdm.npy'))
    ppphhhdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_ppphhhdm.npy'))
    hhhpppdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_hhhpppdm.npy'))
    t1 = ppphhhdm + hhhpppdm

    t_opdm = Tensor(phdm, name='ck')
    t_tpdm = Tensor(pphhdm, name='cckk')
    t_t1 = Tensor(t1, name='t1')

    rdms = MultiTensor([t_opdm, t_tpdm, t_t1])
    db = d2_to_t1(dim)
    rdms.dual_basis = db
    A, _, b = rdms.synthesize_dual_basis()
    primal_vec = rdms.vectorize_tensors()
    residual = csc_matrix(A.dot(primal_vec) - b)
    residual.eliminate_zeros()
    assert np.allclose(residual.toarray(), 0.0)

    phdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_phdm.npy'))
    dim = phdm.shape[0]

    # these are openfermion ordering.
    pphhdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_pphhdm.npy'))
    ppphhhdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_ppphhhdm.npy'))
    hhhpppdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_hhhpppdm.npy'))
    t1 = ppphhhdm + hhhpppdm

    t_opdm = Tensor(phdm, name='ck')
    t_tpdm = Tensor(pphhdm, name='cckk')
    t_t1 = Tensor(t1, name='t1')

    rdms = MultiTensor([t_opdm, t_tpdm, t_t1])
    db = d2_to_t1(dim)
    rdms.dual_basis = db
    A, _, b = rdms.synthesize_dual_basis()
    primal_vec = rdms.vectorize_tensors()
    residual = csc_matrix(A.dot(primal_vec) - b)
    residual.eliminate_zeros()
    assert np.allclose(residual.toarray(), 0.0)


@pytest.mark.skip(reason="A Very expensive test")
def test_t1_construction_iterator():
    """
    Test if we map to the T1 matrix properly
    """
    phdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_phdm.npy'))
    dim = phdm.shape[0]
    # these are openfermion ordering.
    pphhdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_pphhdm.npy'))
    ppphhhdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_ppphhhdm.npy'))
    hhhpppdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_hhhpppdm.npy'))
    t1 = ppphhhdm + hhhpppdm

    t_opdm = Tensor(phdm, name='ck')
    t_tpdm = Tensor(pphhdm, name='cckk')
    t_t1 = Tensor(t1, name='t1')

    rdms = MultiTensor([t_opdm, t_tpdm, t_t1])
    db = d2_to_t1_from_iterator(dim)
    rdms.dual_basis = db
    A, _, b = rdms.synthesize_dual_basis()
    primal_vec = rdms.vectorize_tensors()
    residual = csc_matrix(A.dot(primal_vec) - b)
    residual.eliminate_zeros()
    assert np.allclose(residual.toarray(), 0.0)
    print("HERE IN ITERATOR! YAY")


    phdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_phdm.npy'))
    dim = phdm.shape[0]
    # these are openfermion ordering.
    pphhdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_pphhdm.npy'))
    ppphhhdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_ppphhhdm.npy'))
    hhhpppdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_hhhpppdm.npy'))
    t1 = ppphhhdm + hhhpppdm

    t_opdm = Tensor(phdm, name='ck')
    t_tpdm = Tensor(pphhdm, name='cckk')
    t_t1 = Tensor(t1, name='t1')

    rdms = MultiTensor([t_opdm, t_tpdm, t_t1])
    db = d2_to_t1_from_iterator(dim)
    rdms.dual_basis = db
    A, _, b = rdms.synthesize_dual_basis()
    primal_vec = rdms.vectorize_tensors()
    residual = csc_matrix(A.dot(primal_vec) - b)
    residual.eliminate_zeros()
    assert np.allclose(residual.toarray(), 0.0)
    print("HERE IN ITERATOR! YAY")


@pytest.mark.skip(reason="A Very expensive test")
def test_t1_matrix_construction_iterator():
    """
    Test if we map to the T1 matrix properly
    """
    phdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_phdm.npy'))
    dim = phdm.shape[0]
    pphhdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_pphhdm.npy'))
    ppphhhdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_ppphhhdm.npy'))
    hhhpppdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_hhhpppdm.npy'))
    t1 = (ppphhhdm + hhhpppdm).reshape((dim**3, dim**3))

    bas_elements = []
    for p, q, r in product(range(dim), repeat=3):
        bas_elements.append((p, q, r))
    bas = index_tuple_basis(bas_elements)

    t_opdm = Tensor(phdm, name='ck')
    t_tpdm = Tensor(pphhdm, name='cckk')
    t_t1 = Tensor(t1, name='t1', basis=bas)

    rdms = MultiTensor([t_opdm, t_tpdm, t_t1])
    db = d2_to_t1_matrix(dim)
    rdms.dual_basis = db
    A, _, b = rdms.synthesize_dual_basis()
    primal_vec = rdms.vectorize_tensors()
    residual = csc_matrix(A.dot(primal_vec) - b)
    residual.eliminate_zeros()
    assert np.allclose(residual.toarray(), 0.0)

    phdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_phdm.npy'))
    dim = phdm.shape[0]
    pphhdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_pphhdm.npy'))
    ppphhhdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_ppphhhdm.npy'))
    hhhpppdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_hhhpppdm.npy'))
    t1 = (ppphhhdm + hhhpppdm).reshape((dim**3, dim**3))

    bas_elements = []
    for p, q, r in product(range(dim), repeat=3):
        bas_elements.append((p, q, r))
    bas = index_tuple_basis(bas_elements)

    t_opdm = Tensor(phdm, name='ck')
    t_tpdm = Tensor(pphhdm, name='cckk')
    t_t1 = Tensor(t1, name='t1', basis=bas)

    rdms = MultiTensor([t_opdm, t_tpdm, t_t1])
    db = d2_to_t1_matrix(dim, phdm, pphhdm, t1)
    rdms.dual_basis = db
    A, _, b = rdms.synthesize_dual_basis()
    primal_vec = rdms.vectorize_tensors()
    residual = csc_matrix(A.dot(primal_vec) - b)
    residual.eliminate_zeros()
    assert np.allclose(residual.toarray(), 0.0)

    print("HERE MATRIX for He2H2! YAY")


@pytest.mark.skip(reason="A Very expensive test")
def test_t1_matrix_antisymm():
    """
    Test if we map to the T1 matrix properly
    """
    phdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_phdm.npy'))
    dim = phdm.shape[0]
    pphhdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_pphhdm.npy'))
    ppphhhdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_ppphhhdm.npy'))
    hhhpppdm = np.load(os.path.join(RDM_DIRECTORY, 'h4_rdms/h4_hhhpppdm.npy'))
    t1 = (ppphhhdm + hhhpppdm).reshape((dim**3, dim**3))
    basis_transform = antisymmetry_adapting(dim)
    t1 = basis_transform.T.dot(t1).dot(basis_transform)

    bas_elements = []
    for p, q, r in product(range(dim), repeat=3):
        if p < q < r:
            bas_elements.append((p, q, r))
    bas = index_tuple_basis(bas_elements)

    t_opdm = Tensor(phdm, name='ck')
    t_tpdm = Tensor(pphhdm, name='cckk')
    t_t1 = Tensor(t1, name='t1', basis=bas)

    rdms = MultiTensor([t_opdm, t_tpdm, t_t1])
    db = d2_to_t1_matrix_antisym(dim)
    rdms.dual_basis = db
    A, _, b = rdms.synthesize_dual_basis()
    primal_vec = rdms.vectorize_tensors()
    residual = csc_matrix(A.dot(primal_vec) - b)
    residual.eliminate_zeros()
    assert np.allclose(residual.toarray(), 0.0)

    phdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_phdm.npy'))
    dim = phdm.shape[0]
    pphhdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_pphhdm.npy'))
    ppphhhdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_ppphhhdm.npy'))
    hhhpppdm = np.load(os.path.join(RDM_DIRECTORY, 'he2h2_rdms/he2h2_hhhpppdm.npy'))
    t1 = (ppphhhdm + hhhpppdm).reshape((dim**3, dim**3))
    basis_transform = antisymmetry_adapting(dim)
    t1 = basis_transform.T.dot(t1).dot(basis_transform)

    bas_elements = []
    for p, q, r in product(range(dim), repeat=3):
        if p < q < r:
            bas_elements.append((p, q, r))
    bas = index_tuple_basis(bas_elements)

    t_opdm = Tensor(phdm, name='ck')
    t_tpdm = Tensor(pphhdm, name='cckk')
    t_t1 = Tensor(t1, name='t1', basis=bas)

    rdms = MultiTensor([t_opdm, t_tpdm, t_t1])
    db = d2_to_t1_matrix_antisym(dim)
    rdms.dual_basis = db
    A, _, b = rdms.synthesize_dual_basis()
    primal_vec = rdms.vectorize_tensors()
    residual = csc_matrix(A.dot(primal_vec) - b)
    residual.eliminate_zeros()
    assert np.allclose(residual.toarray(), 0.0)


@pytest.mark.skip(reason="A Very expensive test")
def test_t1_matrix_antiysmm_expensive():
    phdm = np.load(os.path.join(RDM_DIRECTORY, 'lih_rdms/lih_phdm.npy'))
    dim = phdm.shape[0]
    pphhdm = np.load(os.path.join(RDM_DIRECTORY, 'lih_rdms/lih_pphhdm.npy'))
    ppphhhdm = np.load(os.path.join(RDM_DIRECTORY, 'lih_rdms/lih_ppphhhdm.npy'))
    hhhpppdm = np.load(os.path.join(RDM_DIRECTORY, 'lih_rdms/lih_hhhpppdm.npy'))
    t1 = (ppphhhdm + hhhpppdm).reshape((dim**3, dim**3))
    basis_transform = antisymmetry_adapting(dim)
    t1 = basis_transform.T.dot(t1).dot(basis_transform)

    bas_elements = []
    for p, q, r in product(range(dim), repeat=3):
        if p < q < r:
            bas_elements.append((p, q, r))
    bas = index_tuple_basis(bas_elements)

    t_opdm = Tensor(phdm, name='ck')
    t_tpdm = Tensor(pphhdm, name='cckk')
    t_t1 = Tensor(t1, name='t1', basis=bas)

    rdms = MultiTensor([t_opdm, t_tpdm, t_t1])
    db = d2_to_t1_matrix_antisym(dim)
    rdms.dual_basis = db
    A, _, b = rdms.synthesize_dual_basis()
    primal_vec = rdms.vectorize_tensors()
    residual = csc_matrix(A.dot(primal_vec) - b)
    residual.eliminate_zeros()
    assert np.allclose(residual.toarray(), 0.0)
