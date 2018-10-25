"""
Generates an SDP object with cvxpy and solve
"""
import cvxpy as cvx
import numpy as np
from representability.fermions.hamiltonian import spin_adapted_interaction_tensor_rdm_consistent
from representability.fermions.hamiltonian import spin_orbital_interaction_tensor
from representability.fermions.constraints.antisymm_sz_constraints import sz_adapted_linear_constraints
from representability.multitensor import MultiTensor
from representability.tensor import Tensor
from representability.fermions.basis_utils import geminal_spin_basis
from representability.fermions.constraints.spin_orbital_constraints import spin_orbital_linear_constraints


def get_var_indices(tensor, element):
    """
    generate matrix coordinate given the tensor and element reference

    :param Tensor tensor: Tensor object from representability
    :param element: tuple of elements corresponding to the coordinate in
                    tensor form.  For a matrix tensor element will be (a, b)
                    corresponding to the (row, col) index.  For a four index
                    tensor element will be (a, b, c, d) and the corresponding
                    c-ordered element will be returned
    :return: matrix coordinate
    """
    vector_index = tensor.index_vectorized(*element)
    lin_dim = int(np.sqrt(tensor.size))
    return (vector_index // lin_dim, vector_index % lin_dim)


def v2rdm_cvvxpy(one_body_ints, two_body_ints, Na, Nb):
    """
    Generate a cvxpy Problem corresponding the variational 2-RDM method

    Note: if you are using with integrals generated from OpenFermion
    you need to reorder the 2-body ints before passing to this function.
    this can be accomplished by changing the integrals using einsum
    np.einsum('ijkl->ijlk', two_body_ints). the integrals can be found after
    grabbing the hamiltonian from molecule.get_molecular_hamiltonian() and calling
    the attribute `two_body_tensor`.

    This routine returns a cvx problem corresponding to the v2-RDM DQG sdp.
    This can be solved by calling cvx.Problem.solve().  We recommend using
    the SCS solver.

    :param one_body_ints: spinless Fermion one-body integrals
    :param two_body_ints: spinless Fermion basis two-body integrals
    :param Int Na: number of Fermions with alpha-spin
    :param Int Nb: number of Fermions with beta-spin
    :return: cvxpy Problem object, variable dictionary
    :rtype: cvx.Problem
    """
    dim = one_body_ints.shape[0] // 2
    mm = dim ** 2
    mn = dim * (dim - 1) // 2

    h1a, h1b, v2aa, v2bb, v2ab = spin_adapted_interaction_tensor_rdm_consistent(two_body_ints,
                                                                               one_body_ints)
    print("constructing dual basis")
    dual_basis = sz_adapted_linear_constraints(dim, Na, Nb,
                                               ['ck', 'cckk', 'kkcc', 'ckck'])
    print("dual basis constructed")

    bas_aa, bas_ab = geminal_spin_basis(dim)

    v2ab.data *= 2.0

    copdm_a = h1a
    copdm_b = h1b
    coqdm_a = Tensor(np.zeros((dim, dim)), name='kc_a')
    coqdm_b = Tensor(np.zeros((dim, dim)), name='kc_b')
    ctpdm_aa = v2aa
    ctpdm_bb = v2bb
    ctpdm_ab = v2ab
    ctqdm_aa = Tensor(np.zeros((mn, mn)), name='kkcc_aa', basis=bas_aa)
    ctqdm_bb = Tensor(np.zeros((mn, mn)), name='kkcc_bb', basis=bas_aa)
    ctqdm_ab = Tensor(np.zeros((mm, mm)), name='kkcc_ab', basis=bas_ab)

    cphdm_ab = Tensor(np.zeros((dim, dim, dim, dim)), name='ckck_ab')
    cphdm_ba = Tensor(np.zeros((dim, dim, dim, dim)), name='ckck_ba')
    cphdm_aabb = Tensor(np.zeros((2 * mm, 2 * mm)), name='ckck_aabb')

    ctensor = MultiTensor([copdm_a, copdm_b, coqdm_a, coqdm_b, ctpdm_aa, ctpdm_bb, ctpdm_ab,
                           ctqdm_aa, ctqdm_bb, ctqdm_ab, cphdm_ab, cphdm_ba, cphdm_aabb])

    ctensor.dual_basis = dual_basis
    print('synthesizing dual basis')
    # A, _, b = ctensor.synthesize_dual_basis()
    print("dual basis synthesized")


    # create all the psd-matrices for the
    variable_dictionary = {}
    for tensor in ctensor.tensors:
        linear_dim = int(np.sqrt(tensor.size))
        variable_dictionary[tensor.name] = cvx.Variable(shape=(linear_dim, linear_dim), PSD=True, name=tensor.name)

    print("constructing constraints")
    constraints = []
    for dbe in dual_basis:
        single_constraint = []
        for tname, v_elements, p_coeffs in dbe:
            active_indices = get_var_indices(ctensor.tensors[tname], v_elements)
            # vec_idx = ctensor.tensors[tname].index_vectorized(*v_elements)
            # dim = int(np.sqrt(ctensor.tensors[tname].size))
            single_constraint.append(variable_dictionary[tname][active_indices] * p_coeffs)
        constraints.append(cvx.sum(single_constraint) == dbe.dual_scalar)
    print('constraints constructed')

    print("constructing the problem")
    # construct the problem variable for cvx
    objective = cvx.Minimize(
                cvx.trace(copdm_a.data * variable_dictionary['ck_a']) +
                cvx.trace(copdm_b.data * variable_dictionary['ck_b']) +
                cvx.trace(v2aa.data * variable_dictionary['cckk_aa']) +
                cvx.trace(v2bb.data * variable_dictionary['cckk_bb']) +
                cvx.trace(v2ab.data * variable_dictionary['cckk_ab']))

    cvx_problem = cvx.Problem(objective, constraints=constraints)
    print('problem constructed')
    return cvx_problem, variable_dictionary


if __name__ == "__main__":
    # solve the spin problem
    import sys
    from openfermion.hamiltonians import MolecularData
    from openfermionpsi4 import run_psi4
    from openfermion.utils import map_one_pdm_to_one_hole_dm, map_two_pdm_to_two_hole_dm, map_two_pdm_to_particle_hole_dm
    from openfermion.transforms import jordan_wigner
    from representability.fermions.utils import get_molecule_openfermion
    from representability.fermions.constraints.test_antisymm_sz_constraints import system
    from representability.fermions.density.spin_density import SpinOrbitalDensity

    print('Running System Setup')
    basis = 'sto-3g'
    multiplicity = 1
    # charge = 0
    # geometry = [('H', [0.0, 0.0, 0.0]), ('H', [0, 0, 0.75])]
    # charge = 1
    # geometry = [('H', [0.0, 0.0, 0.0]), ('He', [0, 0, 0.75])]
    charge = 0
    geometry = [('H', [0.0, 0.0, 0.0]), ('H', [0, 0, 0.75]),
                ('H', [0.0, 0.0, 2 * 0.75]), ('H', [0, 0, 3 * 0.75])]
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    # Run Psi4.
    molecule = run_psi4(molecule,
                        run_scf=True,
                        run_mp2=False,
                        run_cisd=False,
                        run_ccsd=False,
                        run_fci=True,
                        delete_input=True)

    print('nuclear_repulsion', molecule.nuclear_repulsion)
    print('gs energy ', molecule.fci_energy)
    nuclear_repulsion = molecule.nuclear_repulsion
    gs_energy = molecule.fci_energy

    tpdm = np.einsum('ijkl->ijlk', molecule.fci_two_rdm)
    opdm = molecule.fci_one_rdm
    oqdm = map_one_pdm_to_one_hole_dm(opdm)
    tqdm = map_two_pdm_to_two_hole_dm(tpdm, opdm)
    phdm = map_two_pdm_to_particle_hole_dm(tpdm, opdm)

    tpdm = Tensor(tpdm, name='cckk')
    tqdm = Tensor(tqdm, name='kkcc')
    opdm = Tensor(opdm, name='ck')
    phdm = Tensor(phdm, name='ckck')

    hamiltonian = molecule.get_molecular_hamiltonian()
    one_body_ints, two_body_ints = hamiltonian.one_body_tensor, hamiltonian.two_body_tensor
    two_body_ints = np.einsum('ijkl->ijlk', two_body_ints)

    n_electrons = molecule.n_electrons
    print('n_electrons', n_electrons)
    Na = n_electrons // 2
    Nb = n_electrons // 2

    dim = one_body_ints.shape[0]
    mm = dim ** 2

    h1, v2 = spin_orbital_interaction_tensor(two_body_ints, one_body_ints)
    dual_basis = spin_orbital_linear_constraints(dim, Na + Nb,
                                                 ['ck', 'cckk', 'kkcc', 'ckck'])
    print("constructed dual basis")
    copdm = h1
    coqdm = Tensor(np.zeros((dim, dim)), name='kc')
    ctpdm = v2
    ctqdm = Tensor(np.zeros((dim, dim, dim, dim)), name='kkcc')
    cphdm = Tensor(np.zeros((dim, dim, dim, dim)), name='ckck')

    ctensor = MultiTensor([copdm, coqdm, ctpdm, ctqdm, cphdm])
    ctensor.dual_basis = dual_basis
    print("size of dual basis", len(dual_basis.elements))

    # create all the psd-matrices for the
    variable_dictionary = {}
    for tensor in ctensor.tensors:
        linear_dim = int(np.sqrt(tensor.size))
        variable_dictionary[tensor.name] = cvx.Variable(shape=(linear_dim, linear_dim), PSD=True, name=tensor.name)

    print("constructing constraints")
    constraints = []
    for dbe in dual_basis:
        single_constraint = []
        for tname, v_elements, p_coeffs in dbe:
            active_indices = get_var_indices(ctensor.tensors[tname], v_elements)
            single_constraint.append(variable_dictionary[tname][active_indices] * p_coeffs)
        constraints.append(cvx.sum(single_constraint) == dbe.dual_scalar)
    print('constraints constructed')

    print("constructing the problem")
    # construct the problem variable for cvx
    # interaction_integral_matrix = np.einsum('ijkl->ijlk', v2.data).reshape((dim**2, dim**2))
    interaction_integral_matrix = v2.data.reshape((dim**2, dim**2))

    objective = cvx.Minimize(
                cvx.trace(copdm.data * variable_dictionary['ck']) +
                cvx.trace(interaction_integral_matrix * variable_dictionary['cckk']))

    cvx_problem = cvx.Problem(objective, constraints=constraints)
    print('problem constructed')

    one_energy = np.trace(copdm.data.dot(opdm.data))
    two_energy = np.trace(interaction_integral_matrix @ tpdm.data.reshape((dim**2, dim**2)))  # np.einsum('ijkl,ijkl', tpdm.data, ctpdm.data)
    print(one_energy + two_energy + nuclear_repulsion)

    cvx_problem.solve(solver=cvx.SCS, verbose=True)
    print(cvx_problem.value + nuclear_repulsion, gs_energy)
    # this should give something close to -2.147170020986181
    # assert np.isclose(cvx_problem.value + nuclear_repulsion, gs_energy)  # for 2-electron systems only
    assert np.isclose(cvx_problem.value + nuclear_repulsion, -2.147170020986181, rtol=1.0E-4)

