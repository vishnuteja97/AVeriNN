import interval
import numpy as np
import LP
from Interval import Imatrix, to_Imatrix
from copy import deepcopy


class IStarSet:
    """
    expected types->
    center: interval vector of center, type:Imatrix, shape = (n, 1)
    V: [v1 v2 ... vm], v_i are column vectors, type: Imatrix
    If C*/alpha <= d, then P_coef := C. P_ub := d
    """

    def __init__(self, c, V, P_coef, P_ub):
        if not isinstance(c, Imatrix):
            raise TypeError(f'c passed should be of type \'Imatrix\' but instead got {type(c)}')
        if not isinstance(V, Imatrix):
            raise TypeError(f'V passed should be of type \'Imatrix\' but instead got {type(V)}')

        self.center = c
        self.V = V
        self.P_coef = P_coef
        self.P_ub = P_ub
        self.num_V = len(V)
        self.num_neurons = len(c)

    def print_star(self):
        print('Center:')
        print(self.center)
        print('V:')
        print(self.V)
        print('P_coef')
        print(self.P_coef)
        print('P_ub')
        print(self.P_ub)


def reach_star(inn, inp_stars, method='feasibility'):
    """inp__stars: list of StarSet objects"""
    wts = inn.get_weights()
    bias = inn.get_bias()
    num_layers = len(bias)
    out_stars = inp_stars
    num_stars = []
    for l in range(num_layers - 1):
        out_stars = AffineMap(out_stars, wts[l], bias[l])
        print(f'Affine Layer {l + 1}, IStars: {len(out_stars)}')
        out_stars = ReLUMap(out_stars, method)
        print(f'ReLU Layer {l + 1}, IStars: {len(out_stars)}')
        num_stars.append(len(out_stars))

    out_stars = AffineMap(out_stars, wts[num_layers - 1], bias[num_layers - 1])
    print(f'Affine layer {num_layers - 1}, Stars: {len(out_stars)}')

    return out_stars, num_stars


def AffineMap(stars, W, b):
    """Returns the Affine map operation on a set of stars"""
    out_stars = []

    for star in stars:
        out_stars.append(AffineMapStar(star, W, b))

    return out_stars


def AffineMapStar(star, W, b):
    """ Computes the output of a single input starset across a FNN layer of weights"""

    out_center = (W * star.center) + b
    out_V = W * star.V

    out_star = IStarSet(out_center, out_V, star.P_coef, star.P_ub)

    return out_star


def to_starset(l_bs, u_bs):
    """
    :param l_bs: numpy array of Lower bounds
    :param u_bs: numpy array of corresponding upper bounds
    :return: A star set covering the region specified by the bounds
    Note: 1>= \alpha >= 0
    """

    num_neurons = len(l_bs)

    diff = u_bs - l_bs

    V = np.diag(diff)

    P_coef_ubs = np.identity(num_neurons)
    P_coef_lbs = -1 * np.identity(num_neurons)

    P_coef = np.vstack((P_coef_ubs, P_coef_lbs))

    P_ub = np.concatenate((np.ones(num_neurons), np.zeros(num_neurons)))

    V = to_Imatrix(V.tolist())

    center = to_Imatrix((l_bs.reshape((num_neurons, 1))).tolist())

    star = IStarSet(center, V, P_coef, P_ub)

    return star


def ReLUMap(stars, method='feasibility'):
    """Returns the RELU operation on a set of stars"""
    out_stars = []
    for star in stars:
        relu_stars = ReLUMapSingleStar(star, method)
        out_stars = out_stars + relu_stars

    return out_stars


def ReLUMapSingleStar(star, method='feasibility'):
    """ Returns the RELU operation on a single star, method = {'feasibility', 'approx'}"""
    if method == 'feasibility':
        return ReLUMapSingleStar_feasibility(star)
    elif method == 'approx':
        pass
    else:
        print('Incorrect method name')


def ReLUMapSingleStar_feasibility(star):
    """ Returns the RELU operation on a single star by 'feasibility' method """
    out_stars = [star]

    for i in range(star.num_neurons):
        out_stars = stepRElu_feasibility(i, out_stars)

    return out_stars


def stepRElu_feasibility(neuron, inter_stars):
    """ Returns the stepRElu operation for a set of intermediate stars over a particular neuron """
    out_stars = []
    for star in inter_stars:
        single_star_steprelu = stepREluSingleStar_feasibility(neuron, star)
        out_stars = out_stars + single_star_steprelu

    return out_stars


def stepREluSingleStar_feasibility(neuron, inp_star):
    [star1, star2] = splitStar(neuron, inp_star)
    out_stars = []
    if check_star_feasibility(star1):
        out_stars.append(star1)

    if check_star_feasibility(star2):
        for j in range(len(star2.center.matrix[neuron])):
            star2.center.matrix[neuron][j] = interval.interval(0)
        for j in range(len(star2.V.matrix[neuron])):
            star2.V.matrix[neuron][j] = interval.interval(0)

        out_stars.append(star2)

    return out_stars


def splitStar(neuron, inp_star):
    """
    :param neuron: integer
    :param inp_star: StarSet object
    :return: a list of 2 output stars,  star1: inp_star ^ x_neuron >= 0, star2: inp_star ^ x_neuron <=0
    """
    out_c1 = deepcopy(inp_star.center)
    out_V1 = deepcopy(inp_star.V)
    out_c2 = deepcopy(inp_star.center)
    out_V2 = deepcopy(inp_star.V)

    constraint_low = inp_star.V.get_lower(neuron)
    low_const = inp_star.center.get_lower(neuron)
    constraint_high = inp_star.V.get_upper(neuron)
    high_const = inp_star.center.get_upper(neuron)

    out_P_coef1 = np.vstack([inp_star.P_coef, -1*constraint_low])
    out_P_ub1 = np.append(inp_star.P_ub, low_const)

    out_P_coef2 = np.vstack([inp_star.P_coef, constraint_high])
    out_P_ub2 = np.append(inp_star.P_ub, -1 * high_const)

    star1 = IStarSet(out_c1, out_V1, out_P_coef1, out_P_ub1)  # inp_star ^ x_neuron >=0

    star2 = IStarSet(out_c2, out_V2, out_P_coef2, out_P_ub2)  # inp_star ^ x_neuron <= 0

    return [star1, star2]


def check_star_feasibility(star):
    """Return True if the star set is non-empty"""
    return LP.isFeasible_star(star.P_coef, star.P_ub)
