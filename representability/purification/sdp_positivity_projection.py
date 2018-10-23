import sys
from itertools import product
import numpy as np

from sdpsolve.sdp.sdp import SDP
from sdpsolve.utils.matreshape import vec2block
from sdpsolve.solvers.bpsdp import solve_bpsdp


from representability.fermions.constraints.antisymm_sz_constraints import sz_adapted_linear_constraints, d2_e2_mapping
from representability.fermions.constraints.spin_orbital_constraints import spin_orbital_linear_constraints, \
                                                                           d2_e2_mapping as d2_e2_mapping_spinorbital
from representability.fermions.basis_utils import geminal_spin_basis
from representability.fermions.hamiltonian import spin_orbital_marginal_norm_min

from representability.tensor import Tensor
from representability.multitensor import MultiTensor


def sdp_nrep_sz_reconstruction(corrupted_tpdm_aa, corrupted_tpdm_bb,
                               corrupted_tpdm_ab, num_alpha, num_beta,
                               disp=False, inner_iter_type='EXACT', epsilon=1.0E-8,
                               max_iter=5000):
    if np.ndim(corrupted_tpdm_aa) != 2:
        raise TypeError("corrupted_tpdm_aa must be a 2-tensor")
    if np.ndim(corrupted_tpdm_bb) != 2:
        raise TypeError("corrupted_tpdm_bb must be a 2-tensor")
    if np.ndim(corrupted_tpdm_ab) != 2:
        raise TypeError("corrupted_tpdm_ab must be a 2-tensor")

    if num_alpha != num_beta:
        raise ValueError("right now we are not supporting differing spin numbers")

    spatial_basis_rank = int(np.sqrt(corrupted_tpdm_ab.shape[0]))
    # get basis bijection
    bij_bas_aa, bij_bas_ab = geminal_spin_basis(spatial_basis_rank)

    # build basis look up table
    bas_aa = {}
    bas_ab = {}
    cnt_aa = 0
    cnt_ab = 0
    # iterate over spatial orbital indices
    for p, q in product(range(spatial_basis_rank), repeat=2):
        if q > p:
            bas_aa[(p, q)] = cnt_aa
            cnt_aa += 1
        bas_ab[(p, q)] = cnt_ab
        cnt_ab += 1

    dual_basis = sz_adapted_linear_constraints(spatial_basis_rank, num_alpha, num_beta,
                                               ['ck', 'cckk', 'kkcc', 'ckck'])
    dual_basis += d2_e2_mapping(spatial_basis_rank, bas_aa, bas_ab,
                                corrupted_tpdm_aa, corrupted_tpdm_bb, corrupted_tpdm_ab)

    c_cckk_me_aa = spin_orbital_marginal_norm_min(corrupted_tpdm_aa.shape[0], tensor_name='cckk_me_aa')
    c_cckk_me_bb = spin_orbital_marginal_norm_min(corrupted_tpdm_bb.shape[0], tensor_name='cckk_me_bb')
    c_cckk_me_ab = spin_orbital_marginal_norm_min(corrupted_tpdm_ab.shape[0], tensor_name='cckk_me_ab')
    copdm_a = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank)), name='ck_a')
    copdm_b = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank)), name='ck_b')
    coqdm_a = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank)), name='kc_a')
    coqdm_b = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank)), name='kc_b')
    ctpdm_aa = Tensor(np.zeros_like(corrupted_tpdm_aa), name='cckk_aa', basis=bij_bas_aa)
    ctpdm_bb = Tensor(np.zeros_like(corrupted_tpdm_bb), name='cckk_bb', basis=bij_bas_aa)
    ctpdm_ab = Tensor(np.zeros_like(corrupted_tpdm_ab), name='cckk_ab', basis=bij_bas_ab)
    ctqdm_aa = Tensor(np.zeros_like(corrupted_tpdm_aa), name='kkcc_aa', basis=bij_bas_aa)
    ctqdm_bb = Tensor(np.zeros_like(corrupted_tpdm_bb), name='kkcc_bb', basis=bij_bas_aa)
    ctqdm_ab = Tensor(np.zeros_like(corrupted_tpdm_ab), name='kkcc_ab', basis=bij_bas_ab)

    cphdm_ab = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank, spatial_basis_rank, spatial_basis_rank)), name='ckck_ab')
    cphdm_ba = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank, spatial_basis_rank, spatial_basis_rank)), name='ckck_ba')
    cphdm_aabb = Tensor(np.zeros((2 * spatial_basis_rank**2, 2 * spatial_basis_rank**2)), name='ckck_aabb')

    ctensor = MultiTensor([copdm_a, copdm_b, coqdm_a, coqdm_b, ctpdm_aa, ctpdm_bb, ctpdm_ab, ctqdm_aa, ctqdm_bb,
                           ctqdm_ab, cphdm_ab, cphdm_ba, cphdm_aabb, c_cckk_me_aa, c_cckk_me_bb, c_cckk_me_ab])

    ctensor.dual_basis = dual_basis
    A, _, b = ctensor.synthesize_dual_basis()

    nc, nv = A.shape
    nnz = A.nnz

    sdp = SDP()

    sdp.nc = nc
    sdp.nv = nv
    sdp.nnz = nnz
    sdp.blockstruct = list(map(lambda x: int(np.sqrt(x.size)), ctensor.tensors))
    sdp.nb = len(sdp.blockstruct)
    sdp.Amat = A.real
    sdp.bvec = b.todense().real

    sdp.cvec = ctensor.vectorize_tensors().real

    sdp.Initialize()

    sdp.epsilon = float(epsilon)
    sdp.epsilon_inner = float(epsilon)
    sdp.inner_solve = inner_iter_type
    sdp.disp = disp
    sdp.iter_max = max_iter

    solve_bpsdp(sdp)

    rdms_solution = vec2block(sdp.blockstruct, sdp.primal)
    return rdms_solution[4], rdms_solution[5], rdms_solution[6]


def sdp_nrep_reconstruction(corrupted_tpdm, num_alpha, num_beta):
    """
    Reconstruct a 2-RDm that looks like the input corrupted tpdm

    This reconstruction scheme uses the spin-orbital reconstruction code which is not the optimal size SDP

    :param corrupted_tpdm: measured 2-RDM from the device
    :param num_alpha: number of alpha spin electrons
    :param num_beta: number of beta spin electrons
    :return: purified 2-RDM
    """
    if np.ndim(corrupted_tpdm) != 4:
        raise TypeError("corrupted_tpdm must be a 4-tensor")

    if num_alpha != num_beta:
        raise ValueError("right now we are not supporting differing spin numbers")

    sp_dim = corrupted_tpdm.shape[0]  # single-particle rank
    opdm = np.zeros((sp_dim, sp_dim), dtype=int)
    oqdm = np.zeros((sp_dim, sp_dim), dtype=int)
    tpdm = np.zeros_like(corrupted_tpdm)
    tqdm = np.zeros_like(corrupted_tpdm)
    tgdm = np.zeros_like(corrupted_tpdm)
    opdm = Tensor(tensor=opdm, name='ck')
    oqdm = Tensor(tensor=oqdm, name='kc')
    tpdm = Tensor(tensor=tpdm, name='cckk')
    tqdm = Tensor(tensor=tqdm, name='kkcc')
    tgdm = Tensor(tensor=tgdm, name='ckck')
    error_matrix = spin_orbital_marginal_norm_min(sp_dim ** 2, tensor_name='cckk_me')
    rdms = MultiTensor([opdm, oqdm, tpdm, tqdm, tgdm, error_matrix])

    db = spin_orbital_linear_constraints(sp_dim, num_alpha + num_beta, ['ck', 'cckk', 'kkcc', 'ckck'])
    db += d2_e2_mapping_spinorbital(sp_dim, corrupted_tpdm)

    rdms.dual_basis = db
    A, _, c = rdms.synthesize_dual_basis()
    nv = A.shape[1]
    nc = A.shape[0]
    nnz = A.nnz

    blocklist = [sp_dim, sp_dim, sp_dim ** 2, sp_dim ** 2, sp_dim ** 2, 2 * sp_dim ** 2]
    nb = len(blocklist)

    sdp = SDP()

    sdp.nc = nc
    sdp.nv = nv
    sdp.nnz = nnz
    sdp.blockstruct = blocklist
    sdp.nb = nb
    sdp.Amat = A.real
    sdp.bvec = c.todense().real

    sdp.cvec = rdms.vectorize_tensors().real

    sdp.Initialize()

    sdp.epsilon = float(1.0E-8)
    sdp.inner_solve = "EXACT"
    sdp.disp = True
    solve_bpsdp(sdp)

    solution_rdms = vec2block(blocklist, sdp.primal)
    tpdm_reconstructed = np.zeros_like(corrupted_tpdm)
    for p, q, r, s in product(range(sp_dim), repeat=4):
        tpdm_reconstructed[p, q, r, s] = solution_rdms[2][p * sp_dim + q, r * sp_dim + s]

    return tpdm_reconstructed
