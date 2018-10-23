"""
Construct the DQG conditions were D and Q antisymmetry are taken care of
by
"""
import numpy as np
from itertools import product


def find_num_constraints(mapper, M, sz_adapt, s2_adapt):
    """
    Find the number of constraints implied my the matrices in the mapper

    :param mapper: mapper object relating matrices to variable index
    :param M: number of orbitals (modes)
    :param Bool sz_adapt: turn on Sz operator
    :param Bool s2_adapt: turn on s2_operator
    :return: integer corresponding to the number of linear constraints in the SDP
    :rtype: Int
    """
    m = M
    mm = m*m
    constraint_count = 0
    if sz_adapt:
        constraint_count = 4  # trace on each D2
    else:
        constraint_count = 1

    # antisymmetry constraints on D2(aa), D2(bb)
    if isinstance(mapper['cckk'], dict):
        mbar = m*(m + 1)/2
        constraint_count += 2*mbar*(mbar + 1)/2  # 2*mm*(mm + 1)/2
        # print "antisymmetry constraint", constraint_count
    else:
        mbar = m*(m - 1)/2
        constraint_count += mbar*(mbar + 1)/2

    if 'ck' in mapper.keys():
        if isinstance(mapper['ck'], dict):
            constraint_count += 2*m*(m + 1)/2  # contraction from D2 -> D1
            constraint_count += 2*m*(m + 1)/2  # D1a -> Q1a, D1b -> Q1b
        else:
            constraint_count += m*(m + 1)/2  # contract from D2 -> D1
            constraint_count += m*(m + 1)/2  # D1 -> Q1

    if 'kkcc' in mapper.keys():
        if isinstance(mapper['kkcc'], dict):
            constraint_count += 4*mm*(mm + 1)/2  # D2sigma, sigma -> Q2sigma, sigma
        else:
            constraint_count += mm*(mm + 1)/2

    if 'ckck' in mapper.keys():
        if isinstance(mapper['ckck'], dict):
            # G(aa, aa), {G(aa, bb), G(bb, aa)}, G(bb, bb), G(ab, ab), G(ba, ba)
            constraint_count += 4*mm*(mm + 1)/2  + mm**2
        else:
            constraint_count += mm*(mm + 1)/2

    if 'ME' in mapper.keys():
        if isinstance(mapper['ME'], dict):
            constraint_count += 4*mm*(mm + 1)/2  # I constraint in aa, bb, ab, ba
            constraint_count += 4*mm*(mm + 1)/2  # E constraints in aa, bb, ab, ba
        else:
            constraint_count += 2 * mm*(mm + 1)/2  # I consraint and E constraint

    if 'MR' in mapper.keys():
        if isinstance(mapper['MR'], dict):
            constraint_count += 4*mm*(mm + 1)/2  # I constraint in aa, bb, ab, ba
            constraint_count += 4*mm*(mm + 1)/2  # R constraints in aa, bb, ab, ba
        else:
            constraint_count += 2 * mm*(mm + 1)/2  # I consraint and R constraint

    # print "Total Constraints", constraint_count
    return constraint_count

def trace_d2(A, b, mapper, cnt, M, Na, Nb):
    """
    Generate trace constraints on D2
    """
    for i, j in product(range(M), repeat=2):
        A[cnt, mapper['cckk']['aa'](i, j, i, j)] += 1.0
    b[cnt, 0] = Na*(Na - 1)
    cnt += 1

    for i, j in product(range(M), repeat=2):
        A[cnt, mapper['cckk']['bb'](i, j, i, j)] += 1.0
    b[cnt, 0] = Nb*(Nb - 1)
    cnt += 1

    for i, j in product(range(M), repeat=2):
        A[cnt, mapper['cckk']['ab'](i, j, i, j)] += 1.0
    b[cnt, 0] = Na*Nb
    cnt += 1

    for i, j in product(range(M), repeat=2):
        A[cnt, mapper['cckk']['ba'](i, j, i, j)] += 1.0
    b[cnt, 0] = Na*Nb
    cnt += 1

    return A, b, cnt

def antisymmetric_d2(A, b, mapper, cnt, M):
    """
    Generate antisymmetry constraints on D2aa and D2bb

    Since we are using matrix support with no antisymmeterization we need to add the symmetries
    """
    for p, q, r, s in product(range(M), repeat=4):
        if p*M + q <= r*M + s:
            if p <= q and r <= s:
                A[cnt, mapper['cckk']['aa'](p, q, r, s)] += 0.5
                A[cnt, mapper['cckk']['aa'](r, s, p, q)] += 0.5
                # first negative: flip p, q
                A[cnt, mapper['cckk']['aa'](q, p, r, s)] += 0.5
                A[cnt, mapper['cckk']['aa'](r, s, q, p)] += 0.5
                # second negative: flip s, r
                A[cnt, mapper['cckk']['aa'](p, q, s, r)] += 0.5
                A[cnt, mapper['cckk']['aa'](s, r, p, q)] += 0.5
                # third positive: flip s, r, q, p
                A[cnt, mapper['cckk']['aa'](q, p, s, r)] += 0.5
                A[cnt, mapper['cckk']['aa'](s, r, q, p)] += 0.5
                b[cnt, 0] = 0.0
                cnt += 1

                A[cnt, mapper['cckk']['bb'](p, q, r, s)] += 0.5
                A[cnt, mapper['cckk']['bb'](r, s, p, q)] += 0.5
                # first negative: flip p, q
                A[cnt, mapper['cckk']['bb'](q, p, r, s)] += 0.5
                A[cnt, mapper['cckk']['bb'](r, s, q, p)] += 0.5
                # second negative: flip s, r
                A[cnt, mapper['cckk']['bb'](p, q, s, r)] += 0.5
                A[cnt, mapper['cckk']['bb'](s, r, p, q)] += 0.5
                # third positive: flip s, r, q, p
                A[cnt, mapper['cckk']['bb'](q, p, s, r)] += 0.5
                A[cnt, mapper['cckk']['bb'](s, r, q, p)] += 0.5
                b[cnt, 0] = 0.0
                cnt += 1

    return A, b, cnt

def contract_d2_d1(A, b, mapper, cnt, M, Na, Nb):
    """
    Contractin D2 to D1
    """
    for key in ['ab', 'ba']:  # mapper['cckk'].keys():
        if key == 'ab':
            normalization = Nb
        elif key == 'ba':
            normalization = Na
        elif key == 'aa':
            normalization = Na - 1
        elif key == 'bb':
            normalization = Nb - 1
        else:
            raise KeyError("Normalization Key not recognized")

        for i in range(M):
            for j in range(i, M):
                # contract over r
                for r in range(M):
                    A[cnt, mapper['cckk'][key](i, r, j, r)] += 0.5
                    A[cnt, mapper['cckk'][key](j, r, i, r)] += 0.5

                # add the D1 values
                A[cnt, mapper['ck'][key[0]](i, j)] += -0.5*normalization
                A[cnt, mapper['ck'][key[0]](j, i)] += -0.5*normalization

                # set the b value
                b[cnt, 0] = 0
                cnt += 1

    return A, b, cnt

def d1_to_q1(A, b, mapper, cnt, M):
    """
    Constraints for d1 to q1
    """
    for key in mapper['ck'].keys():
        for i in range(M):
            for j in range(i, M):
                # hermetian constraints
                if i != j:
                    A[cnt, mapper['ck'][key](i, j)] += 0.5
                    A[cnt, mapper['ck'][key](j, i)] += 0.5
                    A[cnt, mapper['kc'][key](j, i)] += 0.5
                    A[cnt, mapper['kc'][key](i, j)] += 0.5
                    b[cnt, 0] = 0.0
                else:
                    A[cnt, mapper['ck'][key](i, j)] += 1.0
                    A[cnt, mapper['kc'][key](j, i)] += 1.0
                    b[cnt, 0] = 1.0

                cnt += 1

    return A, b, cnt


def d2_to_q2(A, b, mapper, cnt, M):
    """
    Constraints for D2 to Q2
    """
    krond = np.eye(M)
    def d2q2map(p, q, r, s, A, b, cnt, M, key, factor=1):
        if key == 'aa':
            key1 = 'a'
            key2 = 'a'
        elif key == 'bb':
            key1 = 'b'
            key2 = 'b'
        elif key == 'ab':
            key1 = 'a'
            key2 = 'b'
        elif key == 'ba':
            key1 = 'b'
            key2 = 'a'

        A[cnt, mapper['ck'][key1](p, r)] += krond[q, s] * factor
        A[cnt, mapper['ck'][key2](q, s)] += krond[p, r] * factor
        if key == 'ab' or key == 'ba':
            pass

        else:
            A[cnt, mapper['ck'][key1](p, s)] -= krond[q, r] * factor
            A[cnt, mapper['ck'][key2](q, r)] -= krond[p, s] * factor
            b[cnt, 0] += -krond[q, r]*krond[p, s] * factor

        b[cnt, 0] += krond[q, s]*krond[p, r] * factor
        A[cnt, mapper['kkcc'][key](r, s, p, q)] += 1.0 * factor
        A[cnt, mapper['cckk'][key](p, q, r, s)] -= 1.0 * factor

    for key in mapper['cckk'].keys():
        for p, q, r, s in product(range(M), repeat=4):
            if p*M + q <= r*M + s:
                d2q2map(p, q, r, s, A, b, cnt, M, key, factor=0.5)
                d2q2map(r, s, p, q, A, b, cnt, M, key, factor=0.5)
                cnt += 1

    return A, b, cnt

def d2_to_g2(A, b, mapper, cnt, M):
    """
    Constraints for D2 to G2
    """
    krond = np.eye(M)
    def g2d2map(p, q, r, s, A, b, cnt, M, key, factor=1):
        """
        Accept pqrs of G2 and map to D2
        """
        quad = {'aaaa': [0, 0], 'bbbb': [1, 1], 'aabb': [0, 1], 'bbaa': [1, 0]}
        if key in ['aaaa', 'bbbb']:
            A[cnt, mapper['ck'][key[0]](p, r)] += krond[q, s] * factor
            A[cnt, mapper['cckk'][key[:2]](p, s, r, q)] += -1.0 * factor
            A[cnt, mapper['ckck']['s_aabb'](p*M + q + quad[key][0]*M**2,
                                            r*M + s + quad[key][1]*M**2)] += -1.0 * factor
            b[cnt, 0] = 0.0

        elif key in ['abab', 'baba']:
            A[cnt, mapper['ck'][key[0]](p, r)] += krond[q, s] * factor
            A[cnt, mapper['cckk'][key[:2]](p, s, r, q)] += -1.0 * factor
            A[cnt, mapper['ckck'][key[:2]](p, q, r, s)] += -1.0 * factor
            b[cnt, 0] = 0.0


        elif key == 'aabb':
            A[cnt, mapper['cckk'][key[0]+key[2]](p, s, q, r)] += 0.5 * factor
            A[cnt, mapper['cckk'][key[2]+key[0]](r, q, s, p)] += 0.5 * factor

            A[cnt, mapper['ckck']['s_aabb'](p*M + q + quad[key][0]*M**2,
                                            r*M + s + quad[key][1]*M**2)] += -0.5 * factor
            A[cnt, mapper['ckck']['s_aabb'](r*M + s + quad['bbaa'][0]*M**2,
                                            p*M + q + quad['bbaa'][1]*M**2)] += -0.5 * factor

            b[cnt, 0] = 0.0


    # note: We are leaving off bbaa block because this gets rolled into
    # aabb block by making hermetian constraints
    for key in ['abab', 'baba', 'aaaa', 'bbbb']:  #
    # for key in mapper['cckk'].keys():
        for p, q, r, s in product(range(M), repeat=4):
            if p*M + q <= r*M + s:
                g2d2map(p, q, r, s, A, b, cnt, M, key, factor=0.5)
                g2d2map(r, s, p, q, A, b, cnt, M, key, factor=0.5)
                # print cnt
                cnt += 1
    for key in ['aabb']:
        for p, q, r, s in product(range(M), repeat=4):
            g2d2map(p, q, r, s, A, b, cnt, M, key, factor=1.0)
            cnt += 1

    return A, b, cnt

def d2_to_me(A, b, mapper, cnt, M):
    """
    Constraints for D2 to ME
    """
    def identity_map(p, q, r, s, A, b, cnt, M, key, factor=1.0):
        if p*M + q != r*M + s:
            A[cnt, mapper['ME'][key](p*M + q, r*M + s)] += 1.0 * factor
            b[cnt, 0] += 0.0
        else:
            A[cnt, mapper['ME'][key](p*M + q, r*M + s)] += 1.0 * factor
            b[cnt, 0] += 1.0 * factor

    def d2_error_map(p, q, r, s, A, b, cnt, M, key, measured_tensors, factor=1.0):
        A[cnt, mapper['ME'][key](p*M + q, r*M + s + M**2)] -= 0.5 * factor
        A[cnt, mapper['ME'][key](p*M + q + M**2, r*M + s)] -= 0.5 * factor
        A[cnt, mapper['cckk'][key](p, q, r, s)] += 1.0 * factor
        b[cnt, 0] += measured_tensors[key][p, q, r, s] * factor


    # first constrain the identity block
    for key in mapper['ME'].keys():
        for p, q, r, s in product(range(M), repeat=4):
            if p*M + q <= r*M + s:
                identity_map(p, q, r, s, A, b, cnt, M, key, factor=0.5)
                identity_map(r, s, p, q, A, b, cnt, M, key, factor=0.5)
                cnt += 1

                d2_error_map(p, q, r, s, A, b, cnt, M, key, measured_tensors, 0.5)
                d2_error_map(r, s, p, q, A, b, cnt, M, key, measured_tensors, 0.5)
                cnt += 1

    return A, b, cnt

def linear_constraints(Na, Nb, M, mapper, sz_adapt=True, s2_adapt=False, measured_tensors=None):
    """
    Construct all trace constraints

    :param Int Na: number of alpha electrons
    :param Int Nb: number of beta electrons
    :param Int M: number of spin-orbitals
    :param mapper: list of mapper objects.
    :param Bool sz_adapt: spin adapt using Sz
    :param Bool s2_adapt: spin adapt using S2
    """
    # first two M*(M + 1)/2 are the D1 -> Q1, D2 -> D1
    # constant 1 is for trace constraint on D2
    # Each of the (M**2)*(M**2 + 1)/2 are for D2 -> Q2, D2->G2, Identity block
    # of ME, difference blocks in ME
    if sz_adapt:
        M = M/2
    mm = M*(M + 1)/2
    N = (Na + Nb).real

    num_constraints = find_num_constraints(mapper, M if sz_adapt else M, sz_adapt, s2_adapt)
    A = np.zeros((num_constraints, mapper.max_index))
    b = np.zeros((num_constraints, 1))

    # set constraint counter
    cnt = 0

    # trace on D2
    A, b, cnt = trace_d2(A, b, mapper, cnt, M, Na, Nb)

    # antisymmetry
    A, b, cnt = antisymmetric_d2(A, b, mapper, cnt, M)

    if 'ck' in mapper.keys():
        # contraction from D2 to D1
        A, b, cnt = contract_d2_d1(A, b, mapper, cnt, M, Na, Nb)

        # Now map D1 to Q1
        A, b, cnt = d1_to_q1(A, b, mapper, cnt, M)


    ################################################
    #
    # The following constraints are symmeterized
    #
    ################################################

    # d2 -> q2
    krond = np.eye(M)
    if 'kkcc' in mapper.keys():
        A, b, cnt = d2_to_q2(A, b, mapper, cnt, M)

    # d2 -> g2
    if "ckck" in mapper.keys():
        A, b, cnt = d2_to_g2(A, b, mapper, cnt, M)

    if "ME" in mapper.keys():
        if measured_tensors is None:
            raise ValueError("You must give me the 2-RDM you've measured!")

        A, b, cnt = d2_to_me(A, b, mapper, cnt, M)

    return A, b


def generate_block_list(mapper, M):
    """
    Generate the list of linear sizes for the blocks of the SDP

    """
    # print "here"
    block_list = []
    if 'ck' in mapper.keys():
        block_list.append(mapper['ck']['a'].index_map.shape[0])
        block_list.append(mapper['ck']['b'].index_map.shape[0])
        block_list.append(mapper['kc']['a'].index_map.shape[0])
        block_list.append(mapper['kc']['b'].index_map.shape[0])

    if 'cckk' in mapper.keys():
        block_list.append(mapper['cckk']['aa'].index_map.shape[0]**2)
        block_list.append(mapper['cckk']['bb'].index_map.shape[0]**2)
        block_list.append(mapper['cckk']['ab'].index_map.shape[0]**2)
        if 'ba' in mapper['cckk'].keys():
            block_list.append(mapper['cckk']['ba'].index_map.shape[0]**2)

    if 'kkcc' in mapper.keys():
        block_list.append(mapper['kkcc']['aa'].index_map.shape[0]**2)
        block_list.append(mapper['kkcc']['bb'].index_map.shape[0]**2)
        block_list.append(mapper['kkcc']['ab'].index_map.shape[0]**2)
        if 'ba' in mapper['cckk'].keys():
            block_list.append(mapper['cckk']['ba'].index_map.shape[0]**2)

    if 'ckck' in mapper.keys():
        block_list.append(mapper['ckck']['s_aabb'].index_map.shape[0])
        block_list.append(mapper['ckck']['ab'].index_map.shape[0]**2)
        block_list.append(mapper['ckck']['ba'].index_map.shape[0]**2)

    if 'ME' in mapper.keys():
        block_list.extend([2*M**2]*4)

    return block_list
