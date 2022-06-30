import LP
import numpy as np

def is_star_safe2(star, spec):
    constraint_coefs = np.array(spec[0])
    bound_consts = np.array(spec[1])

    extra_P_coef = constraint_coefs @ star.V
    extra_P_ub = bound_consts - (constraint_coefs @ star.center.reshape((star.num_neurons, 1))).flatten()

    intersection_P_coef = np.vstack([star.P_coef, extra_P_coef])
    intersection_P_ub = np.concatenate([star.P_ub, extra_P_ub])

    f_check = LP.isFeasible_star(intersection_P_coef, intersection_P_ub)

    if f_check is True:
        return False  # UNSAFE
    else:
        return True  # SAFE


def is_star_safe(star):
    """
    Checks by intersecting star with the (unsafe)polyhedron
    x0 <= x1; x0 <= x2; x0 <= x3; x0 <= x4
    If intersection feasible -> unsafe star.
    Implemented by adding 4 additional constraints to alpha vector
    """
    if len(star.center) != 5 or len(star.V) != 5:
        raise StarOutputSizeMismatch(len(star.center))

    extra_P_coef = np.array([star.V[0] - star.V[i] for i in [1, 2, 3, 4]])
    extra_P_ub = np.array([star.center[i] - star.center[0] for i in [1, 2, 3, 4]])

    intersection_P_coef = np.vstack([star.P_coef, extra_P_coef])
    intersection_P_ub = np.concatenate([star.P_ub, extra_P_ub])

    f_check = LP.isFeasible_star(intersection_P_coef, intersection_P_ub)

    if f_check is True:
        return False  # UNSAFE
    else:
        return True  # SAFE


def tag_check(stars, index):
    max_ind = len(stars[0].tag)

    while index < max_ind:
        P = [s for s in stars if s.tag[index] == 'P']
        N = [s for s in stars if s.tag[index] == 'N']

        if P and N:
            tag_check(P, index + 1)
            tag_check(N, index + 1)

        index = index + 1


def verify_output(stars, specs):
    for star in stars:
        for spec in specs:
            if not is_star_safe2(star, spec):
                return 'sat'

    return 'unsat'
