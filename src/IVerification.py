import LP
import numpy as np
from Interval import Imatrix


def is_star_safe(star, spec):
    """
    Given a constraint matrix Ay <= B, where 'y' is the output vector
    We generate the interval constraints as:
    A*star.center + (A*star.V).alpha <= B

    The above are equivalent to the below linear constraints(when alpha >= 0)
    sup(A*star.V).alpha <= B - sup(A*star.center)
    """
    constraint_coefs = spec[0]  # A
    bound_consts = spec[1]  # B

    # Converting constraint matrix to an interval matrix in order to multiply with interval matrices
    con_coefs_Imat = Imatrix(constraint_coefs, constraint_coefs)

    extra_P_coef = (con_coefs_Imat * star.V).upper
    extra_P_ub = bound_consts - (con_coefs_Imat * star.center).upper.flatten()

    intersection_P_coef = np.vstack([star.P_coef, extra_P_coef])
    intersection_P_ub = np.concatenate([star.P_ub, extra_P_ub])

    f_check = LP.isFeasible_star(intersection_P_coef, intersection_P_ub)

    if f_check is True:
        return False  # UNSAFE
    else:
        return True  # SAFE


def verify_output(stars, specs):
    for star in stars:
        for spec in specs:
            if not is_star_safe(star, spec):
                return 'sat'

    return 'unsat'
