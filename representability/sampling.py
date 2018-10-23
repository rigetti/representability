from collections import Counter
from itertools import product

import numpy as np
from pyquil.gates import H, RX, I
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.quil import Program
# from referenceqvm.api import QVMConnection
from referenceqvm.unitary_generator import tensor_up


def add_gaussian_noise(tensor, sqrt_variance):
    """
    Iterate over tensor and apply Gaussian noise

    :param tensor:
    :param M:
    :param variance:
    :return:
    """
    corrupted_tensor = np.copy(tensor)
    if np.isclose(sqrt_variance, 0.0):
        return corrupted_tensor

    for indices in product(range(tensor.shape[0]), repeat=tensor.ndim):
        corrupted_tensor[indices] += np.random.normal(0, scale=sqrt_variance)

    return corrupted_tensor


def add_gaussian_noise_antisymmetric_four_tensor(tensor, std_error):
    """
    Iterate over tensor and apply Gaussian noise

    :param tensor:
    :param M:
    :param std_error:
    :return:
    """
    corrupted_tensor = np.copy(tensor)
    if np.isclose(std_error, 0.0):
        return corrupted_tensor

    dim = tensor.shape[0]
    for p, q, r, s in product(range(dim), repeat=4):
        if p * dim + q >= r * dim + s and p < q and r < s:
            # the normal distribution used by numpy, scale variable is the dtandard error
            # which is the square root of the variance
            corrupted_tensor[p, q, r, s] += np.random.normal(0, scale=std_error)
            corrupted_tensor[q, p, r, s] = -corrupted_tensor[p, q, r, s]
            corrupted_tensor[p, q, s, r] = -corrupted_tensor[p, q, r, s]
            corrupted_tensor[q, p, s, r] = corrupted_tensor[p, q, r, s]

            corrupted_tensor[r, s, p, q] = corrupted_tensor[p, q, r, s]
            corrupted_tensor[r, s, q, p] = corrupted_tensor[q, p, r, s]
            corrupted_tensor[s, r, p, q] = corrupted_tensor[p, q, s, r]
            corrupted_tensor[s, r, q, p] = corrupted_tensor[q, p, s, r]

    return corrupted_tensor


def string_2_pauli(pauli_string):
    string_length = len(pauli_string)
    if string_length % 2 != 0:
        raise TypeError("Pauli Terms are even string lenght")

    pauli_op = PauliTerm("I", 0)
    for term in range(string_length / 2):
        pauli_op = pauli_op * PauliTerm(pauli_string[term * 2],
                                        int(pauli_string[2 * term + 1]))

    return pauli_op


def rotate_density(rho, pauli_term, debug=False):
    """
    Rotate the density so I can read off in the computational basis
    """
    # rotate operator into computational basis
    rot_prog = Program()
    n_qubits = int(np.log2(rho.shape[0]))

    marked_qubits = []
    for key, value in pauli_term._ops.iteritems():
        marked_qubits.append(key)
        if value == "X":
            rot_prog.inst(H(key))
        elif value == "Y":
            rot_prog.inst(RX(np.pi / 2)(key))
    rot_prog.inst(I(n_qubits - 1))

    qvm_unitary = QVMConnection(type_trans='unitary')
    if debug:
        ham_op = tensor_up(PauliSum([pauli_term]), n_qubits)
        e_true = np.trace(ham_op.dot(rho))

    unitary = qvm_unitary.unitary(rot_prog)
    rho = unitary.dot(rho.dot(np.conj(unitary).T))

    if debug:
        z_term = PauliTerm("I", 0)
        for idx in marked_qubits:
            z_term = z_term * PauliTerm("Z", idx)

        ham_op_2 = tensor_up(pauli_term.coefficient * PauliSum([z_term]),
                             n_qubits)
        test_expect = np.trace(np.dot(ham_op_2, rho))
        assert np.isclose(test_expect, e_true)

    return rho, marked_qubits


def sample_pauli_base(rho, pauli_term, epsilon, base_shots=5000, debug=False,
                      disp=True):
    """
    Sample using the inverse transform sampling technique

    :param rho: density matrix of state.
    :param pauli_term: operator to measure.
    :param epsilon: absolute precision.  The standard error of the mean of the
                    expected value will be sampled until this value is achieved.
    :param base_shots: (Optional. Default=5000). If there is a low variance
                       in the operator then many shots might be required to get
                       to a non-zero sample variance.  This sets the minimum
                       number of samples for operator averaging.
    :param disp: (Optional. Default=True) Display the mean and standard error
                 as sampling occurs.
    """
    if not isinstance(pauli_term, PauliTerm):
        if not isinstance(pauli_term, str):
            raise TypeError("pauli_term must be a pauli term id or a PauliTerm")
        else:
            try:
                pauli_term = string_2_pauli(pauli_term)
            except:
                raise ValueError("Could not translate string to pauliterm")

    n_qubits = int(np.log2(rho.shape[0]))
    rho, marked_qubits = rotate_density(rho, pauli_term, debug=debug)

    cdf = np.cumsum(np.diag(rho))
    num_qubits = int(np.log2(np.shape(rho)[0]))

    # # now compare the calculation with an update of themean and variance
    term_counter = 0
    var_sum = 0
    var_square = 0
    a_k = 0.0
    a_k_1 = 0.0
    q_k = 0.0
    eps = 1000
    while eps > epsilon or term_counter < base_shots:
        u = np.random.random()
        state_index = np.searchsorted(cdf, u)
        if state_index == len(cdf):
            state_index = state_index - 1
        bit_result = map(int, np.binary_repr(state_index, width=num_qubits))
        bit_result_int = int("".join([str(x) for x in bit_result]), 2)
        binary = int(parity_even_p(bit_result_int, marked_qubits))

        term_counter += 1
        var_sum += binary
        var_square += 1

        a_k = a_k_1 + (binary - a_k_1) / (term_counter)
        q_k = q_k + (binary - a_k_1) * (binary - a_k)

        # update sample mean and variance
        if term_counter < 100:
            sample_variance = 1
        else:
            sample_variance = q_k / (term_counter - 1)

        # update mean and epsilon
        eps = np.sqrt((pauli_term.coefficient ** 2) * (
        2 ** 2) * sample_variance / term_counter)
        if disp and term_counter % 5000:
            print('std dev ', eps, " mean ", (a_k * 2 - 1) * pauli_term.coefficient)

        a_k_1 = a_k

    return (a_k * 2.0 - 1.0) * pauli_term.coefficient, eps ** 2, term_counter


def fast_sample(rho, pauli_term, trials=1, debug=False):
    """
    Sample using the inverse transform sampling technique

    :param rho: density matrix of state
    :param pauli_term: operator to measure
    """
    if not isinstance(pauli_term, PauliTerm):
        if not isinstance(pauli_term, str):
            raise TypeError("pauli_term must be a pauli term id or a PauliTerm")
        else:
            try:
                pauli_term = string_2_pauli(pauli_term)
            except:
                raise ValueError("Could not translate string to pauliterm")

    n_qubits = int(np.log2(rho.shape[0]))
    rho, marked_qubits = rotate_density(rho, pauli_term, debug=debug)

    cdf = np.cumsum(np.diag(rho))
    num_qubits = int(np.log2(np.shape(rho)[0]))
    bit_results = []
    for _ in range(trials):
        u = np.random.random()
        state_index = np.searchsorted(cdf, u)
        if state_index == len(cdf):
            state_index = state_index - 1
        bit_results.append(
            map(int, np.binary_repr(state_index, width=num_qubits))[::-1])

    # check if we can calculate expected value
    bit_results_int = map(lambda y: int("".join([str(x) for x in y[::-1]]), 2),
                          bit_results)
    binary = map(lambda x: int(parity_even_p(x, marked_qubits)),
                 bit_results_int)  # true/false +1/0

    binary = np.array(binary)
    signs = (2 * binary - 1)
    values = signs * pauli_term.coefficient
    p1 = np.sum(binary) / float(trials)

    # # now compare the calculation with an update of themean and variance
    term_counter = 0
    var_sum = 0
    var_square = 0
    a_k = 0.0
    a_k_1 = 0.0
    q_k = 0.0
    for term in binary:
        term_counter += 1
        var_sum += term
        var_square += 1

        a_k = a_k_1 + (term - a_k_1) / (term_counter)
        q_k = q_k + (term - a_k_1) * (term - a_k)

        # update sample mean and variance
        if term_counter == 1:
            sample_variance = 0
        else:
            sample_variance = q_k / (term_counter - 1)

        # update mean and epsilon
        # eps = np.sqrt(sample_variance)
        a_k_1 = a_k

    freq = Counter(map(tuple, bit_results))

    # perform weighted average
    expectation = 0
    for bitstring, count in freq.items():
        bitstring_int = int("".join([str(x) for x in bitstring[::-1]]), 2)
        if parity_even_p(bitstring_int, marked_qubits):
            expectation += float(count) / trials
        else:
            expectation -= float(count) / trials


    return expectation * pauli_term.coefficient, np.sqrt(((
                                                          pauli_term.coefficient ** 2) * (
                                                          2 ** 2) * q_k / (
                                                          trials - 1)) / trials)


def run_measure(cdf, num_qubits):
    """
    Inverse sampling method.  Given the cumulative distribution funciton give me a measurement

    https://en.wikipedia.org/wiki/Inverse_transform_sampling
    """
    u = np.random.random()
    state_index = np.searchsorted(cdf, u)
    if state_index == len(cdf):
        state_index = state_index - 1
    return map(int, np.binary_repr(state_index, width=num_qubits))[::-1]


def sample_pauli(pauli_term, epsilon, rho):
    """
    Given a pauli term and an epislon of precion and rho eval the expected value

    :param pauli_term: (Str or PauliTerm)
    :param epsilon: (Float) precesion of standard error
    :param rho: Density matrix to sample from
    :returns: expected value of the pauli term
    :rtype: float
    """
    if not isinstance(pauli_term, PauliTerm):
        if not isinstance(pauli_term, str):
            raise TypeError("pauli_term must be a pauli term id or a PauliTerm")
        else:
            try:
                pauli_term = string_2_pauli(pauli_term)
            except:
                raise ValueError("Could not translate string to pauliterm")

    # get the initial values for sampling 
    rho, marked_qubits = rotate_density(rho, pauli_term)
    num_qubits = int(np.log2(rho.shape[0]))
    cdf = np.cumsum(np.diag(rho))
    term_counter = 0
    var_sum = 0
    var_square = 0
    a_k = 0.0
    a_k_1 = 0.0
    q_k = 0.0
    eps = 1.0
    samples = []
    while eps > epsilon:
        bitstring = run_measure(cdf, num_qubits)
        bitstring_int = int("".join([str(x) for x in bitstring[::-1]]), 2)
        binary = parity_even_p(bitstring_int, marked_qubits)
        samples.append(binary)

        term_counter += 1
        var_sum += binary
        var_square += 1

        a_k = a_k_1 + (binary - a_k_1) / (term_counter)
        q_k = q_k + (binary - a_k_1) * (binary - a_k)

        # update sample mean and variance
        if term_counter == 1:
            sample_variance = 1
        else:
            sample_variance = q_k / (term_counter - 1)

        # update mean and epsilon
        if term_counter < 10:
            eps = 1.0
        else:
            eps = np.sqrt((pauli_term.coefficient ** 2) * (
            2 ** 2) * sample_variance / term_counter)

        a_k_1 = a_k

    assert np.isclose(
        pauli_term.coefficient * (2 * np.mean(np.array(samples)) - 1),
        pauli_term.coefficient * (2 * a_k - 1))
    assert np.isclose(q_k / (term_counter - 1),
                      np.std(np.array(samples), ddof=1) ** 2)
    return pauli_term.coefficient * (2 * a_k - 1), np.sqrt(((
                                                            pauli_term.coefficient ** 2) * (
                                                            2 ** 2) * q_k / (
                                                            term_counter - 1)) / term_counter), term_counter


def parity_even_p(state, marked_qubits):
    """
    Calculates the parity of elements at indexes in marked_qubits

    Parity is relative to the binary representation of the integer state.

    :param state: The wavefunction index that corresponds to this state.
    :param marked_qubits: The indexes to be considered in the parity sum.
    :returns: A boolean corresponding to the parity.
    """
    assert isinstance(state, int), "{} is not an integer. Must call " \
                                   "parity_even_p with an integer " \
                                   "state.".format(state)
    mask = 0
    for q in marked_qubits:
        mask |= 1 << q
    return bin(mask & state).count("1") % 2 == 0


def test_pauli_rotation(ham, n_density):
    expectation = 0.0
    for ham_term in ham.terms.terms:
        expectation += ham_term.coefficient * fast_sample(n_density,
                                                          ham_term.id(),
                                                          debug=True)
    assert np.isclose(expectation.real, gs_eig)


def map_paulis_d2(d2ab_pauli_map, p_set_probs, m):
    d2ab = np.zeros((m, m, m, m), dtype=complex)
    for (p, q, r, s), p_terms in d2ab_pauli_map.iteritems():
        for term in p_terms.terms:
            if p * m + q == r * m + s:
                d2ab[p, q, r, s] += term.coefficient * p_set_probs[term.id()]
            else:
                d2ab[p, q, r, s] += term.coefficient * p_set_probs[term.id()]
                d2ab[r, s, p, q] += term.coefficient * p_set_probs[term.id()]

    return d2ab.real


def count_measured_pterms(pterm):
    """
    count the number terms to measure in a pauli sum
    """
    if not isinstance(pterm, PauliSum):
        raise TypeError("count_measured_pterms() only accepts Pauli Terms")

    count = 0
    for term in pterm.terms:
        if not term.id() == "":
            if np.isclose(term.coefficient.imag, 0.0):
                count += 1
    return count


if __name__ == "__main__":
    from representability.fermions.utils import get_molecule

    ham, ham_op, mol_data, gs_wf, n_density, gs_eig = get_molecule()
    n_qubits = ham.n_qubits
    M = 2 * mol_data.M_
    N = mol_data.nElectrons_

    d2ab_pauli_map, pauli_set = pauliD2set_sz(ham)

    # measure the following pauli term
    # expected value from density 0.002722159645
    # pauli_operator = PauliTerm("X", 1, 0.125)*PauliTerm("Z", 2)*PauliTerm("X", 3)
    pauli_operator = PauliTerm("Z", 2, 0.25) * PauliTerm("Z", 3, 1.0)
    p_op_mat = tensor_up(PauliSum([pauli_operator]), M)
    true_expectation = np.trace(np.dot(p_op_mat, n_density))
    # assert np.isclose(true_expectation, 0.002722159645)A
    assert np.isclose(true_expectation, 0.249863555819)


    for key, value in d2ab_pauli_map.iteritems():
        for pterm in value.terms:
            op = tensor_up(PauliSum([pterm]), n_qubits)
            if np.isclose(pterm.coefficient.real, 0) and not np.isclose(
                    pterm.coefficient.imag, 0.0):
                assert np.isclose(np.trace(np.dot(op, n_density)), 0.0)
