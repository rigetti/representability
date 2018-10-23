import sys
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import os
from math import factorial
from itertools import product

from representability.config import DATA_DIRECTORY
from openfermionpsi4 import run_psi4
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner

from referenceqvm.unitary_generator import tensor_up
from forestopenfermion import qubitop_to_pyquilpauli

from representability.fermions.basis_utils import generate_parity_permutations


def get_molecule_openfermion(molecule, eigen_index=0):
    # check if the molecule is in the molecules data directory
    if molecule.name + '.hdf5' in os.listdir(DATA_DIRECTORY):
        print("\tLoading File from {}".format(DATA_DIRECTORY))
        molecule.load()
    else:
        # compute properties with run_psi4
        molecule = run_psi4(molecule, run_fci=True)
        print("\tPsi4 Calculation Completed")
        print("\tSaved in {}".format(DATA_DIRECTORY))
        molecule.save()

    fermion_hamiltonian = molecule.get_molecular_hamiltonian()
    qubitop_hamiltonian = jordan_wigner(fermion_hamiltonian)
    psum = qubitop_to_pyquilpauli(qubitop_hamiltonian)
    ham = tensor_up(psum, molecule.n_qubits)
    if isinstance(ham, (csc_matrix, csr_matrix)):
        ham = ham.toarray()
    w, v = np.linalg.eigh(ham)
    gs_wf = v[:, [eigen_index]]
    n_density = gs_wf.dot(np.conj(gs_wf).T)
    return molecule, v[:, [eigen_index]], n_density, w[eigen_index]


def wedge_product(tensor_a, tensor_b):
    """
    Returns the antisymmetric tensor product between two tensor operators

    Tensor operators have the same number of upper and lower indices

    :param tensor_a: tensor of p-rank.
    :param tensor_b: tensor of q-rank
    :returns: tensor_a_w_b antisymmetric tensor of p + q rank
    """
    # get the number of upper and lower indices
    rank_a = int(len(tensor_a.shape)/2)
    rank_b = int(len(tensor_b.shape)/2)

    permutations = generate_parity_permutations(rank_a + rank_b)

    # define new tensor product which is the appropriate size
    new_tensor = np.zeros((tensor_a.shape[:rank_a] + tensor_b.shape[:rank_b] +
                           tensor_a.shape[rank_a:] + tensor_b.shape[rank_b:]), dtype=complex)

    for indices in product(*list(map(lambda x: range(x), new_tensor.shape))):
        idx_upper = np.array(indices[:rank_a + rank_b])
        idx_lower = np.array(indices[rank_a + rank_b:])

        # sum over all over permutations and load into element of tensor
        for perm_u, parity_u in permutations:
            for perm_l, parity_l in permutations:

                # index permutation
                a_idx_u = list(idx_upper[perm_u][:rank_a])
                b_idx_u = list(idx_upper[perm_u][rank_a:])
                a_idx_l = list(idx_lower[perm_l][:rank_a])
                b_idx_l = list(idx_lower[perm_l][rank_a:])

                parity_term = parity_u * parity_l
                # sum into new_tensor
                new_tensor[indices] += parity_term * (tensor_a[tuple(a_idx_u + a_idx_l)] * tensor_b[tuple(b_idx_u + b_idx_l)])

    new_tensor /= factorial(rank_a + rank_b)**2
    return new_tensor


def map_d1_q1(opdm, oqdm):
    """
    demonstrate mapping to opdm and oqdm
    """
    m = opdm.shape[0]
    I = np.eye(m)
    for i, j in product(range(m), repeat=2):
        # we will use hermetian constraints here
        assert np.isclose(0.5*(opdm[i, j] + oqdm[j, i] + opdm[i, j] + oqdm[j, i]), I[i, j])


def map_d2_q2(tpdm, tqdm, opdm):
    """
    demonstrate map
    """
    sm_dim = opdm.shape[0]
    krond = np.eye(sm_dim)
    for p, q, r, s in product(range(sm_dim), repeat=4):
        term1 = opdm[p, r]*krond[q, s] + opdm[q, s]*krond[p, r]
        term2 = -1*(opdm[p, s]*krond[r, q] + opdm[q, r]*krond[s, p])
        term3 = krond[s, p]*krond[r, q] - krond[q, s]*krond[r, p]
        term4 = tqdm[r, s, p, q]
        # print tpdm[p, q, r, s], term1 + term2 + term3 + term4
        assert np.isclose(tpdm[p, q, r, s], term1 + term2 + term3 + term4)


def map_d2_q2_antisymm(tpdm, tqdm, opdm, bas):
    """
    demonstrate map
    """
    sm_dim = opdm.shape[0]
    krond = np.eye(sm_dim)
    for p, q, r, s in product(range(sm_dim), repeat=4):
        if p < q and r < s:
            term1 = 2.0 * opdm[p, r]*krond[q, s] + 2.0 * opdm[q, s]*krond[p, r]
            term2 = -1*(2.0 * opdm[p, s]*krond[r, q] + 2.0 * opdm[q, r]*krond[s, p])
            term3 = 2.0 * krond[s, p]*krond[r, q] - 2.0 * krond[q, s]*krond[r, p]
            term4 = tqdm[bas[(r, s)], bas[(p, q)]]
            # print tpdm[bas[(p, q)], bas[(r, s)]], term1 + term2 + term3 + term4
            assert np.isclose(tpdm[bas[(p, q)], bas[(r, s)]], term1 + term2 + term3 + term4)


def map_d2_q2_ab(tpdm, tqdm, opdm_a, opdm_b):
    """
    demonstrate map
    """
    sm_dim = opdm_a.shape[0]
    krond = np.eye(sm_dim)
    for p, q, r, s in product(range(sm_dim), repeat=4):
        term1 = opdm_a[p, r]*krond[q, s] + opdm_b[q, s]*krond[p, r]
        term2 = -krond[p, r]*krond[q, s]
        term3 = tqdm[r, s, p, q]
        assert np.isclose(tpdm[p, q, r, s], term1 + term2 + term3)


def map_d2_q2_ab_mat(tpdm, tqdm, opdm_a, opdm_b):
    """
    demonstrate map
    """
    sm_dim = opdm_a.shape[0]
    krond = np.eye(sm_dim)
    for p, q, r, s in product(range(sm_dim), repeat=4):
        term1 = opdm_a[p, r]*krond[q, s] + opdm_b[q, s]*krond[p, r]
        term2 = -krond[p, r]*krond[q, s]
        term3 = tqdm[r * sm_dim + s, p * sm_dim + q]
        assert np.isclose(tpdm[p * sm_dim + q, r * sm_dim + s], term1 + term2 + term3)


def map_d2_g2(tpdm, tgdm, opdm):
    """
    demonstrate map
    """
    sm_dim = opdm.shape[0]
    krond = np.eye(sm_dim)
    for p, q, r, s in product(range(sm_dim), repeat=4):
        term1 = opdm[p, r]*krond[q, s]
        term2 = -1*tgdm[p, s, r, q]
        assert np.isclose(tpdm[p, q, r, s], term1 + term2)


def map_d2_g2_sz_antisymm(tpdm, tgdm, opdm, bas):
    """
    demonstrate map
    """
    sm_dim = opdm.shape[0]
    krond = np.eye(sm_dim)
    for p, q, r, s in product(range(sm_dim), repeat=4):
        if p < q and r < s:
            term1 = 0.5 * (opdm[p, r]*krond[q, s] + opdm[q, s]*krond[p, r])
            term2 = -0.5 * (opdm[q, r] * krond[p, s] + opdm[p, s] * krond[q, r])
            term3 = -0.5 * (tgdm[p, s, r, q] + tgdm[q, r, s, p])
            term4 = 0.5 * (tgdm[q, s, r, p] + tgdm[p, r, s, q])
            # print tpdm[bas[(p, q)], bas[(r, s)]], term1 + term2 + term3 + term4
            assert np.isclose(tpdm[bas[(p, q)], bas[(r, s)]], term1 + term2 + term3 + term4)


def map_g2_d2_sz_antisymm(tgdm, tpdm_anti, opdm, bas):
    """
    map p, q, r, s of G2 to the elements of D2 antisymmetric
    """
    sm_dim = opdm.shape[0]
    krond = np.eye(sm_dim)

    def compare(p, q, r, s, tgdm, tpdm, opdm, bas, factor=1.0):
        term1 = tgdm[p, q, r, s]
        term2 = 1.0 * opdm[p, r]*krond[q, s]
        if p != s and r != q:
            # print "non-zero term", p, s, r, q, p>s, r > q, (-1)**(p > s), (-1)**(r > q)
            gem1 = tuple(sorted([p, s]))
            gem2 = tuple(sorted([r, q]))
            parity = (-1)**(p > s) * (-1)**(r > q)
            term3 = parity * tpdm[bas[gem1], bas[gem2]] * -0.5
        else:
            # print "zero term"
            term3 = 0.0
        # print term1, term2 + term3
        assert np.isclose(term1, term2 + term3)

    for p, q, r, s in product(range(sm_dim), repeat=4):
        if p * sm_dim + q <= r * sm_dim + s:
            compare(p, q, r, s, tgdm, tpdm_anti, opdm, bas, factor=0.5)


def map_g2ab_ba_d2ab(tgdm_ab, tgdm_ba, tpdm_ab, opdm_a, opdm_b):
    """
    Mapping pieces of G2ab/ba to D2ab
    """
    m_dim = opdm_a.shape[0]
    krond = np.eye(m_dim)

    def compare(p, q, r, s, factor=1.0):
        term1 = tgdm_ab[p, q, r, s] + tgdm_ba[s, r, q, p]
        term2 = opdm_a[p, r] * krond[q, s] + opdm_b[s, q] * krond[p, r]
        term3 = -2.0 * tpdm_ab[p * m_dim + s, r * m_dim + q]
        # print term1, term2 + term3
        assert np.isclose(term1, term2 + term3)

    for p, q, r, s in product(range(m_dim), repeat=4):
        compare(p, q, r, s)


def map_d2_g2_sz_antisymm_aabb(tpdm, tgdm_aabb, tgdm_bbaa, bas, sm_dim):
    """
    demonstrate map
    """
    for p, q, r, s in product(range(sm_dim), repeat=4):
        term1 = tgdm_aabb[p, q, r, s] + tgdm_bbaa[r, s, p, q]
        term2 = -1.0 * (tpdm[bas[(p, s)], bas[(q, r)]] + tpdm[bas[(q, r)], bas[(p, s)]])
        # print term1, -term2
        assert np.isclose(term1, -term2)


def map_d2_g2_sz(tgdm, tpdm, opdm, g2_block='aaaa'):
    sm_dim = opdm.shape[0]
    krond = np.eye(sm_dim)

    if g2_block == 'aaaa' or g2_block == 'bbbb':
        for p, q, r, s in product(range(sm_dim), repeat=4):
            term1 = opdm[p, r]*krond[q, s]
            term2 = -1*tpdm[p, s, r, q]
            # print tgdm[p, q, r, s], term1 + term2
            assert np.isclose(tgdm[p, q, r, s], term1 + term2)
    elif g2_block == 'bbaa' or g2_block == 'aabb':
        for p, q, r, s in product(range(sm_dim), repeat=4):
            term1 = tpdm[p, s, q, r]
            # print tgdm[p, q, r, s,], term1
            assert np.isclose(tgdm[p, q, r, s], term1)
    elif g2_block == 'abab' or g2_block == 'baba':
        for p, q, r, s in product(range(sm_dim), repeat=4):
            term1 = opdm[p, r]*krond[q, s]
            if len(tpdm.shape) == 2:
                term2 = -1*tpdm[p*sm_dim + s, r*sm_dim + q]
            else:
                term2 = -1*tpdm[p, s, r, q]
            assert np.isclose(tgdm[p, q, r, s], term1 + term2)


def map_d2_d1(tpdm, opdm):
    """
    map tpdm to trace corrected versions of 1-RDM
    """
    sm_dim = opdm.shape[0]
    N = np.trace(opdm)
    Na = sum([opdm[2*i, 2*i] for i in range(sm_dim // 2)])
    Nb = sum([opdm[2*i + 1, 2*i + 1] for i in range(sm_dim // 2)])
    for p, q, in product(range(sm_dim), repeat=2):
        term = 0
        for r in range(sm_dim):
            term += tpdm[p, r, q, r]

        assert np.isclose(term/(N - 1), opdm[p, q])

    # also check spin constraints.  Contract just alpha terms and get answers.
    # Contract just beta terms and get answers.

    # print "number of alpha particles "
    # print Na
    # print "number of beta particles "
    # print Nb

    # # first contract alpha electrons
    # print "single particle basis rank"
    # print sm_dim
    # print "alpha opdm"
    for p, q in product(range(sm_dim/2), repeat=2):
        term1 = 0
        for r in range(sm_dim):
            term1 += tpdm[2*p, r, 2*q, r]

        # print term1/(Na + Nb - 1), opdm[2*p, 2*q]
        assert np.isclose(term1/(Na + Nb - 1), opdm[2*p, 2*q])


def map_d2_d1_sz(tpdm, opdm, scalar):
    assert np.allclose(np.einsum('ijkj', tpdm), opdm * scalar)


def map_d2_d1_sz_antisym(tpdm, opdm, scalar, bas):
    test_opdm = np.zeros_like(opdm)
    for i, j in product(range(opdm.shape[0]), repeat=2):
        for r in range(opdm.shape[0]):
            if i != r and j != r:
                top_gem = tuple(sorted([i, r]))
                bot_gem = tuple(sorted([j, r]))
                parity = (-1)**(r < i) * (-1)**(r < j)
                test_opdm[i, j] += tpdm[bas[top_gem], bas[bot_gem]] * 0.5 * parity
        # print test_opdm[i, j] / scalar, opdm[i, j]
    assert np.allclose(test_opdm / scalar, opdm)


def map_d2_d1_sz_symm(tpdm, opdm, scalar, bas):
    test_opdm = np.zeros_like(opdm)
    m = opdm.shape[0]
    for i, j in product(range(opdm.shape[0]), repeat=2):
        for r in range(opdm.shape[0]):
            # top_gem = tuple([i, r])
            # bot_gem = tuple([j, r])
            # print top_gem, bot_gem
            # test_opdm[i, j] += tpdm[bas[top_gem], bas[bot_gem]]
            test_opdm[i, j] += tpdm[i * m + r, j * m + r]

    assert np.allclose(test_opdm, opdm * scalar)


def check_antisymmetric_d2(tpdm):
    m = tpdm.shape[0]
    for p, q, r, s in product(range(m), repeat=4):
        # assert np.isclose(tpdm[p, q, r, s], -1*tpdm[q, p, r, s])
        # assert np.isclose(tpdm[p, q, r, s], -1*tpdm[p, q, s, r])
        # assert np.isclose(tpdm[p, q, r, s], tpdm[q, p, s, r])
        np.testing.assert_almost_equal(tpdm[p, q, r, s], -1*tpdm[q, p, r, s])
        np.testing.assert_almost_equal(tpdm[p, q, r, s], -1*tpdm[p, q, s, r])
        np.testing.assert_almost_equal(tpdm[p, q, r, s], tpdm[q, p, s, r])


def map_d2_e2_sz(tpdm, m_tensor, measured_tensor):
    m = tpdm.shape[0]
    assert np.allclose(m_tensor[:m**2, :m**2], np.eye(m**2))
    # assert np.allclose(m_tensor[m**2:, m**2:], np.zeros((m**2, m**2)))
    for p, q, r, s in product(range(m), repeat=4):
        # print m_tensor[p*m + q + m**2, r*m + s], tpdm[p, q, r, s] - measured_tensor[p, q, r, s]
        assert np.isclose(m_tensor[p*m + q + m**2, r*m + s], tpdm[p, q, r, s] - measured_tensor[p, q, r, s])
        assert np.isclose(m_tensor[r*m + s + m**2, p*m + q], tpdm[r, s, p, q] - measured_tensor[r, s, p, q])
        assert np.isclose(m_tensor[p*m + q, r*m + s + m**2], tpdm[p, q, r, s] - measured_tensor[p, q, r, s])
        assert np.isclose(m_tensor[r*m + s, p*m + q + m**2], tpdm[r, s, p, q] - measured_tensor[r, s, p, q])


def map_d2_e2_sz_antisymm(tpdm, m_tensor, measured_tensor, m, bas):
    tm = m*(m - 1)/2
    assert np.allclose(m_tensor[:tm, :tm], np.eye(tm))
    for p, q, r, s in product(range(m), repeat=4):
        if p < q and r < s:
            assert np.isclose(m_tensor[bas[(p, q)] + tm, bas[(r, s)]], tpdm[bas[(p, q)], bas[(r, s)]] - measured_tensor[bas[(p, q)], bas[(r, s)]])
            assert np.isclose(m_tensor[bas[(r, s)] + tm, bas[(p, q)]], tpdm[bas[(r, s)], bas[(p, q)]] - measured_tensor[bas[(r, s)], bas[(p, q)]])
            assert np.isclose(m_tensor[bas[(p, q)], bas[(r, s)] + tm], tpdm[bas[(p, q)], bas[(r, s)]] - measured_tensor[bas[(p, q)], bas[(r, s)]])
            assert np.isclose(m_tensor[bas[(r, s)], bas[(p, q)] + tm], tpdm[bas[(r, s)], bas[(p, q)]] - measured_tensor[bas[(r, s)], bas[(p, q)]])


def map_d2_e2_sz_symm(tpdm, m_tensor, measured_tensor, m, bas):
    mm = m*m
    assert np.allclose(m_tensor[:mm, :mm], np.eye(mm))
    for p, q, r, s in product(range(m), repeat=4):
        assert np.isclose(m_tensor[bas[(p, q)] + mm, bas[(r, s)]], tpdm[bas[(p, q)], bas[(r, s)]] - measured_tensor[bas[(p, q)], bas[(r, s)]])
        assert np.isclose(m_tensor[bas[(r, s)] + mm, bas[(p, q)]], tpdm[bas[(r, s)], bas[(p, q)]] - measured_tensor[bas[(r, s)], bas[(p, q)]])
        assert np.isclose(m_tensor[bas[(p, q)], bas[(r, s)] + mm], tpdm[bas[(p, q)], bas[(r, s)]] - measured_tensor[bas[(p, q)], bas[(r, s)]])
        assert np.isclose(m_tensor[bas[(r, s)], bas[(p, q)] + mm], tpdm[bas[(r, s)], bas[(p, q)]] - measured_tensor[bas[(r, s)], bas[(p, q)]])


def map_d1_e1_sz_symm(opdm, m_tensor, measured_tensor, m):
    assert np.allclose(m_tensor[:m, :m], np.eye(m))
    for p, q, in product(range(m), repeat=2):
        assert np.isclose(m_tensor[p + m, q], opdm[p, q] - measured_tensor[p, q])
        assert np.isclose(m_tensor[p, q + m], opdm[p, q] - measured_tensor[p, q])


def four_tensor2matrix(tensor):
    dim = tensor.shape[0]
    mat = np.zeros((dim**2, dim**2), dtype=tensor.dtype)
    for p, q, r, s in product(range(dim), repeat=4):
        mat[p*dim + q, r*dim + s] = tensor[p, q, r, s]
    return mat


def matrix2four_tensor(matrix):
    dim = matrix.shape[0]
    sp_dim = int(np.sqrt(dim))
    four_tensor = np.zeros((sp_dim, sp_dim, sp_dim, sp_dim), dtype=matrix.dtype)
    for p, q, r, s in product(range(sp_dim), repeat=4):
        four_tensor[p, q, r, s] = matrix[p * sp_dim + q, r * sp_dim + s]

    return four_tensor
