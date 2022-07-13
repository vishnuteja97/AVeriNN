import LP
import numpy as np
from interval import interval

def is_star_safe(star, spec):
    constraint_coefs = spec[0]
    bound_consts = spec[1]
    num_constraints = len(constraint_coefs)
    num_neurons = star.V.shape[0]
    num_alphas = star.V.shape[1]

    extra_P_coef = []
    extra_P_ub = []
    for constraint_num in range(num_constraints):
        new_P_coef = []
        new_P_ub = 0

        c = interval(0)
        for i in range(num_neurons):
            c = c + (star.center.matrix[i][0] * constraint_coefs[constraint_num][i])

        new_P_ub = (bound_consts[constraint_num] - c[0].sup)

        for alpha in range(num_alphas):
            v_alpha = interval(0)

            for i in range(num_neurons):
                v_alpha = v_alpha + (star.V.matrix[i][alpha] * constraint_coefs[constraint_num][i])

            new_P_coef.append(v_alpha[0].sup)

        extra_P_coef.append(new_P_coef)
        extra_P_ub.append(new_P_ub)

    intersection_P_coef = np.vstack([star.P_coef, np.array(extra_P_coef)])
    intersection_P_ub = np.concatenate([star.P_ub, np.array(extra_P_ub)])

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
