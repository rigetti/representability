"""Generate a density object that has a similar interface as the Density objects
of representability"""
import numpy as np
from itertools import product
from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
from representability.purification.fermionic_marginal import (map_opdm_to_oqdm,
            map_tpdm_to_tqdm, map_tpdm_to_phdm)
from representability.fermions.utils import four_tensor2matrix


class MissingCalculationError(Exception):
    pass


class Psi4SpinOrbitalDensity(object):
    """
    Calculates the rdms from a psi4 calculation.

    This generally is faster and should be used instead of a large tensor_up
    calculation
    """
    def __init__(self, molecule, wf_type='fci'):
        """Take a OpenFermion molecule object and return one with calcs"""
        if not isinstance(molecule, MolecularData):
            raise TypeError("molecule must be an OpenFermion Molecular Data Object")

        # get the calculation type
        run_scf = 1
        run_mp2 = 0
        run_cisd = 0
        run_ccsd = 0
        run_fci = 1
        molecule_psi4 = run_psi4(molecule, run_scf=run_scf, run_mp2=run_mp2,
                                 run_cisd=run_cisd, run_ccsd=run_ccsd,
                                 run_fci=run_fci)

        tpdm_index_exchange = np.einsum('ijkl->ijlk', molecule_psi4.fci_two_rdm)

        self.opdm = molecule_psi4.fci_one_rdm
        self.tpdm = tpdm_index_exchange
        self.oqdm = None
        self.tqdm = None
        self.phdm = None

    def construct_opdm(self):
        """
        Return the one-particle density matrix

        <psi|a_{p}^{\dagger}a_{q}|psi>
        """
        if self.opdm is not None:
            return self.opdm
        else:
            raise MissingCalculationError("opdm was never calculated")

    def construct_ohdm(self):
        """
        Return the one-hole density matrix

        <psi|a_{p}a_{q}^{\dagger}|psi>
        """
        # check if both are not available
        if self.opdm is None and self.oqdm is None:
            raise MissingCalculationError("need an opdm calculated")

        # check if oqdm hasn't been calculated yet
        elif self.oqdm is None and self.opdm is not None:
            self.oqdm = map_opdm_to_oqdm(self.opdm)
            return self.oqdm

        # if both are available then just return
        else:
            return self.oqdm

    def construct_tpdm(self):
        """
        Return the two-particle density matrix

        <psi|a_{p}^{\dagger}a_{q}^{\dagger}a_{s}a_{r}|psi>
        """
        if self.tpdm is None:
            raise MissingCalculationError("need an tpdm calculated")
        else:
            return self.tpdm

    def construct_thdm(self):
        """
        Return the two-hole density matrix

        <psi|a_{p}a_{q}a_{s}^{\dagger}a_{r}^{\dagger}|psi>
        """
        if self.opdm is None or self.tpdm is None:
            raise MissingCalculationError("Need a opdm or a tpdm calculated")

        # i haven't calculated tqdm yet. calc and store
        elif self.tqdm is None and self.tpdm is not None:
            self.tqdm = map_tpdm_to_tqdm(self.tpdm, self.opdm)
            return self.tqdm

        # if both are availabe then just return
        else:
            return self.tqdm

    def construct_phdm(self):
        """
        Return the particle-hole density matrix

        <psi|a_{p}^{\dagger}a_{q}a_{s}^{\dagger}a_{r}|psi>
        """
        if self.opdm is None or self.tpdm is None:
            raise MissingCalculationError("Need a opdm or a tpdm calculated")
        elif self.phdm is None and self.tpdm is not None:
            self.phdm = map_tpdm_to_phdm(self.tpdm, self.opdm)
            return self.phdm
        else:
            return self.phdm

    def construct_tpdm_error_matrix(self, error_tpdm):
        """
        Construct the error tensor

        Structure of tensor is a large matrix with the following block structure

        [I] [E]
        [E] [F]

        where I is the identity matrix, E is ^{2}D_{meas} - ^{2}D_{var}, F is a matrix of Free variables

        :return:
        """

        # if our tensor is rank 4 then we need to reshape
        if np.ndim(error_tpdm) == 4:
            error_tpdm_matrix = four_tensor2matrix(error_tpdm)
        else:
            error_tpdm_matrix = np.copy(error_tpdm)

        dim = error_tpdm_matrix.shape[0]

        top_row_emat = np.hstack((np.eye(dim), error_tpdm_matrix))
        bottom_row_emat = np.hstack((error_tpdm_matrix.T, np.zeros((dim, dim), dtype=error_tpdm.dtype)))
        error_schmidt_matrix = np.vstack((top_row_emat, bottom_row_emat))

        return error_schmidt_matrix


