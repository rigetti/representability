"""This file grabs molecular data and rdms from alternative sources"""
from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4


def get_openfermion_molecule(geometry, basis, charge, multiplicity,
                             calc_type='fci'):
    """
    Get openfermion molecule objects

    :param geometry: list of tuples where first element of the tuple is the
                     atom string, and next three coordinates are XYZ geom
    :param basis: (string) of basis set
    :param multiplicity: eigenvalue of S^{2} (2S + 1)
    :param charge: (int) charge on molecule
    :param calc_type: (string) valid calculation types are:
                      ['scf', 'mp2', 'cisd', 'ccsd', 'fci']. default is 'fci'
                      if 'fci' is selected the fci_one_rdm and fci_two_rdm
                      fields are populated and can be used later.  Conversion
                      to particle and holes can occur with methods in the
                      representability.purification.fermionic_marginal
    :return:
    """
    valid_calc_types = ['scf', 'mp2', 'cisd', 'ccsd', 'fci']
    if calc_type not in valid_calc_types:
        raise TypeError("Calculation type is not valid")

    molecule = MolecularData(geometry, basis, multiplicity, charge)

    # get the calculation type
    run_scf = 1
    run_mp2 = 0
    run_cisd = 0
    run_ccsd = 0
    run_fci = 1
    # default to fci run
    molecule = run_psi4(molecule, run_scf=run_scf, run_mp2=run_mp2, run_cisd=run_cisd,
                        run_ccsd=run_ccsd, run_fci=run_fci)
    return molecule
