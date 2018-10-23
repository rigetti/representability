"""
Set of tools for constructing basis operator maps in canonical ordering

Author: Nicholas C. Rubin
"""
import copy
from itertools import product
import numpy as np
from math import factorial
from scipy.special import comb
from representability.tensor import index_tuple_basis


def geminal_spin_basis(m_dim):
    """
    Construct the geminal basis blocked by Sz

    D2 and Q2 have the same aa, bb, ab basis structure.
    :param Int m_dim: rank of the single particle position space basis (NOT SPIN ORBITAL RANK)
    :return: Basis Bijections
    :rtype: representability.tensor.Bijection
    """
    # basis construction for canonical matrices
    gem_aa_bas = []
    gem_ab_bas = []
    for i in range(m_dim):
        for j in range(m_dim):
            if i < j:
                gem_aa_bas.append((i, j))

            gem_ab_bas.append((i, j))

    return index_tuple_basis(gem_aa_bas), index_tuple_basis(gem_ab_bas)


def triples_spin_orbital_antisymm_basis(m_dim):
    """
    Construct the triples spin basis

    :param Int m_dim:
    :return: representbaility.tensor.Bijection
    """
    bas = []
    for p, q, r in product(range(m_dim), repeat=3):
        if p < q < r:
            bas.append((p, q, r))
    return index_tuple_basis(bas)


def generate_parity_permutations(n):
    """
    Generates the permutations of n indices :-> range(n).

    :param Int n: number of elements to permute
    """
    indices = list(range(1, int(n)))
    permutations = [([0], 1)]
    while len(indices) > 0:
        index_to_inject = indices.pop(0)

        new_permutations = []  # permutations in the tree
        for perm in permutations:
            # now loop over positions to insert
            for put_index in range(len(perm[0]) + 1):
                new_index_list = copy.deepcopy(perm[0])
                # insert new object starting at end of the list
                new_index_list.insert(len(perm[0]) - put_index, index_to_inject)

                new_permutations.append((new_index_list, perm[1] * (-1)**(put_index)))

        permutations = new_permutations

    return permutations


def _coord_generator(i, j, k, l):
    """
    Generator for equivalent spin-orbital indices given a set of four spin-orbitals

    indices are in chemist notation so for real-valued chemical spinless Hamiltonians
    the integrals have 8-fold symmetry.

    using this function and iterating over the following:
        i >= j && k >= l && ij >= kl
        where ij = i*(i + 1)/2 + j
              kl = k*(k + 1)/2 + l

    spatial real-values
    i, j, k, l
    j, i, k, l
    i, j, l, k
    j, i, l, k
    k, l, i, j
    k, l, j, i
    l, k, i, j
    l, k, j, i
    """
    unique_set =  {(i, j, k, l),
                   (j, i, k, l),
                   (i, j, l, k),
                   (j, i, l, k),
                   (k, l, i, j),
                   (k, l, j, i),
                   (l, k, i, j),
                   (l, k, j, i)}
    for index_element in unique_set:
        yield index_element


def _three_parity(p, q, r):
    parity_terms = [(p, q, r, 1),  # no switch
                    (p, r, q, -1),  # r <-> q
                    (q, p, r, -1),  # p <-> q
                    (q, r, p, 1),  # p <-> q, r <-> p
                    (r, p, q, 1),  # r <-> q, r <-> p
                    (r, q, p, -1)  # r <-> q, r <-> p, p <-> q
                    ]
    for term in parity_terms:
        yield term


def antisymmetry_adapting(dim):
    """
    Generate the unitary matrix that transforms the dim**6 to the n choose 3 x
    n choose 3 matrix

    We reverse the order of pp, qq, rr because this is acting on the annihilator
    indices in the 3-RDM.  Each row is indexed by p^ q^ r^ and annihilator is
    indexed by i j k from <p^ q^ r^ i j k> but rember we store
    3D_{ijk}^{pqr} = <p^ q^ r^ k j i>
    this means that if we iterate over i, j, k then the transform corresponds to
    k-j-i index
    :param dim:
    :return: unitary
    """
    t1_dim = int(comb(dim, 3))
    basis_transform = np.zeros((dim ** 3, t1_dim))
    normalization = 1 / np.sqrt(factorial(3))
    # for idx in range(t1_dim):  # column index
    idx = 0
    for i, j, k in product(range(dim), repeat=3):  # row index in each column
        if i < j < k:
            for ii, jj, kk, parity in _three_parity(i, j, k):
                basis_transform[ii * dim**2 + jj * dim + kk, idx] += parity * normalization
            idx += 1

    return basis_transform

