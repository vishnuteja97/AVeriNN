import numpy as np
import LP
import NN


class StarSet:
    """
    expected types->
    center: vector of center, type:np.array(dim=1)
    V: [v1 v2 ... vm], v_i are column vectors, type: np.array(dim=2)
    If C*/alpha <= d, then P_coef := C. P_ub := d
    """

    def __init__(self, c, V, P_coef, P_ub):
        self.center = c
        self.V = V
        self.P_coef = P_coef
        self.P_ub = P_ub
        self.num_V = len(V)
        self.num_neurons = len(c)
        self.tag = []

    def print_star(self):
        print('Center:')
        print(self.center)
        print('V:')
        print(self.V)
        print('P_coef')
        print(self.P_coef)
        print('P_ub')
        print(self.P_ub)


def reach_star(nn, inp_stars, method='bounds'):
    """inp__stars: list of StarSet objects"""
    wts = nn.get_weights()
    bias = nn.get_bias()
    num_layers = len(bias)
    out_stars = inp_stars
    list_of_splits = []
    num_stars = []
    for l in range(num_layers - 1):
        out_stars = AffineMap(out_stars, wts[l], bias[l])
        print(f'Affine Layer {l + 1}, Stars: {len(out_stars)}')
        out_stars, split_set = ReLUMap(out_stars, method)
        print(f'ReLU Layer {l + 1}, Stars: {len(out_stars)}')
        list_of_splits.append(len(split_set))
        num_stars.append(len(out_stars))

    out_stars = AffineMap(out_stars, wts[num_layers - 1], bias[num_layers - 1])
    print(f'Affine layer {num_layers - 1}, Stars: {len(out_stars)}')

    return out_stars, list_of_splits, num_stars


def AffineMap(stars, W, b):
    """Returns the Affine map operation on a set of stars"""
    out_stars = []

    for star in stars:
        out_stars.append(AffineMapStar(star, W, b))

    return out_stars


def AffineMapStar(star, W, b):
    """ Computes the output of a single input starset across a FNN layer of weights"""

    out_center = W.dot(star.center) + b
    out_V = W.dot(star.V)

    out_star = StarSet(out_center, out_V, star.P_coef, star.P_ub)

    out_star.tag = star.tag.copy()

    return out_star


def ReLUMap(stars, method='bounds'):
    """Returns the RELU operation on a set of stars"""
    out_stars = []
    global_split_set = set()
    for star in stars:
        relu_stars, split_set = ReLUMapSingleStar(star, method)
        out_stars = out_stars + relu_stars
        global_split_set = global_split_set.union(split_set)

    print(f'Relu map stars, number of stars is {len(out_stars)}')

    return out_stars, global_split_set


def ReLUMapSingleStar(star, method='feasibility'):
    """ Returns the RELU operation on a single star, method = {'feasibility', 'bounds'}"""
    if method == 'feasibility':
        return ReLUMapSingleStar_feasibility(star)
    elif method == 'bounds':
        return ReLUMapSingleStar_bounds(star)
    elif method == 'approx':
        return ReLUMapSingleStar_approx(star)
    else:
        print('Incorrect method name')


def ReLUMapSingleStar_approx(star):
    """ Returns the RELU operation on a single star by triangle overapproximation of relu """
    out_star = star

    for i in range(star.num_neurons):
        out_star = stepRElu_approx(i, out_star)

    return [out_star]


def stepRElu_approx(neuron, inter_star):
    """ Returns the stepRElu operation for an intermediate star over a particular neuron """
    dim = len(inter_star.center)
    out_star = None
    l, u = get_range(inter_star, neuron)
    k = u / (u - l)

    if l >= 0:
        out_star = inter_star
    elif u <= 0:
        out_star = inter_star
        out_star.center[neuron] = 0
        out_star.V[neuron] = 0
    else:
        out_star = inter_star
        out_star.center[neuron] = 0
        out_star.V[neuron] = 0

        E_neuron = np.zeros((dim, 1))
        E_neuron[neuron] = 1

        # Now to add constraints

        num_of_alphas = out_star.P_coef.shape[1]
        num_rows_P = out_star.P_coef.shape[0]

        out_star.P_coef = np.hstack([out_star.P_coef, np.zeros((num_rows_P, 1))])

        C1 = np.zeros((1, num_of_alphas + 1))
        d1 = 0

        C2 = np.append(out_star.V[neuron], -1)
        d2 = -1 * out_star.center[neuron]

        C3 = np.append(-k * out_star.V[neuron], 1)
        d3 = k * l * (1 - out_star.center[neuron])

        out_star.V = np.hstack((out_star.V, E_neuron))
        out_star.P_coef = np.vstack([out_star.P_coef, C1, C2, C3])
        out_star.P_ub = np.append(out_star.P_ub, [d1, d2, d3])

    return out_star


def ReLUMapSingleStar_feasibility(star):
    """ Returns the RELU operation on a single star by 'feasibility' method """
    out_stars = [star]
    splits = set()

    for i in range(star.num_neurons):
        out_stars, split = stepRElu_feasibility(i, out_stars)
        if split is True:
            splits.add(i)

    return out_stars, splits


def stepRElu_feasibility(neuron, inter_stars):
    """ Returns the stepRElu operation for a set of intermediate stars over a particular neuron """
    out_stars = []
    split_check = False
    for star in inter_stars:
        single_star_steprelu = stepREluSingleStar_feasibility(neuron, star)
        out_stars = out_stars + single_star_steprelu
        if len(single_star_steprelu) == 2:
            split_check = True

    return out_stars, split_check


def stepREluSingleStar_feasibility(neuron, inp_star):
    [star1, star2] = splitStar(neuron, inp_star, relu_map='no')
    star1.tag = inp_star.tag.copy()
    star2.tag = inp_star.tag.copy()
    star1.tag.append('P')
    star2.tag.append('N')
    out_stars = []
    split_check = 0
    if check_star_feasibility(star1):
        out_stars.append(star1)
        split_check = split_check + 1

    if check_star_feasibility(star2):
        star2.center[neuron] = 0
        star2.V[neuron] = 0
        out_stars.append(star2)

    return out_stars


def ReLUMapSingleStar_bounds(star):
    """ Returns the RELU operation on a single star by 'bounds' method """
    Map = find_bounds_map(star)
    out_stars = [star]

    for i in range(star.num_neurons):
        out_stars = stepRElu_bounds(i, out_stars, Map[i])

    return out_stars


def stepRElu_bounds(neuron, inter_stars, map_val):
    """ Returns the stepRElu operation for a set of intermediate stars over a particular neuron """
    out_stars = []
    for star in inter_stars:
        out_stars = out_stars + stepREluSingleStar_bounds(neuron, star, map_val)

    return out_stars


def stepREluSingleStar_bounds(neuron, inp_star, map_val):
    """ Returns the stepRElu operation for a single star over a particular neuron """
    if map_val == 1:
        out_c = np.copy(inp_star.center)
        out_V = np.copy(inp_star.V)
        out_P_coef = np.copy(inp_star.P_coef)
        out_P_ub = np.copy(inp_star.P_ub)

        return [StarSet(out_c, out_V, out_P_coef, out_P_ub)]

    elif map_val == -1:
        out_c = np.copy(inp_star.center)
        out_V = np.copy(inp_star.V)
        out_P_coef = np.copy(inp_star.P_coef)
        out_P_ub = np.copy(inp_star.P_ub)
        out_c[neuron] = 0
        out_V[neuron] = 0

        return [StarSet(out_c, out_V, out_P_coef, out_P_ub)]

    else:
        star1, star2 = splitStar(neuron, inp_star, relu_map='yes')
        is_feas_star1 = check_star_feasibility(star1)
        is_feas_star2 = check_star_feasibility(star2)

        if is_feas_star1:
            if is_feas_star2:
                return [star1, star2]
            else:
                return [star1]
        else:
            if is_feas_star2:
                return [star2]
            else:
                return []


def splitStar(neuron, inp_star, relu_map='no'):
    """
    :param neuron: integer
    :param inp_star: StarSet object
    :param relu_map = Split stars are output by default, on 'yes' input the relu is performed on split stars
    :return: a list of 2 output stars,  star1: inp_star ^ x_neuron >= 0, star2: inp_star ^ x_neuron <=0
    """
    out_c1 = np.copy(inp_star.center)
    out_V1 = np.copy(inp_star.V)
    out_c2 = np.copy(inp_star.center)
    out_V2 = np.copy(inp_star.V)

    constraint1 = -1 * out_V1[neuron]
    constraint2 = out_V2[neuron]

    out_P_coef1 = np.vstack([inp_star.P_coef, constraint1])
    out_P_ub1 = np.append(inp_star.P_ub, inp_star.center[neuron])

    out_P_coef2 = np.vstack([inp_star.P_coef, constraint2])
    out_P_ub2 = np.append(inp_star.P_ub, -1 * inp_star.center[neuron])

    star1 = StarSet(out_c1, out_V1, out_P_coef1, out_P_ub1)  # inp_star ^ x_neuron >=0

    star2 = StarSet(out_c2, out_V2, out_P_coef2, out_P_ub2)  # inp_star ^ x_neuron <= 0
    if relu_map == 'yes':
        star2.center[neuron] = 0
        star2.V[neuron] = 0

    return [star1, star2]


def find_bounds_map(star):
    """
    :param star: A single StarSet object
    :return: A Map(list) of neurons to: 1 if always positive, -1 if always negative, 0 otw
    """
    M = []
    lbs, ubs = output_range(star)

    for i in range(star.num_neurons):
        if lbs[i] >= 0.0:
            M.append(1)
        elif ubs[i] <= 0.0:
            M.append(-1)
        else:
            M.append(0)

    return M


def check_star_feasibility(star):
    """Return True if the star set is non-empty"""
    return LP.isFeasible_star(star.P_coef, star.P_ub)


def output_range(star):
    # Fix error(wrong bounds calculated)
    lower_bounds = np.array([0] * star.num_neurons, dtype='float64')
    upper_bounds = np.array([0] * star.num_neurons, dtype='float64')

    for i in range(star.num_neurons):
        i_lb = LP.optimize_star(star.V[i, :], star.P_coef, star.P_ub, mode='min')
        i_ub = LP.optimize_star(star.V[i, :], star.P_coef, star.P_ub, mode='max')

        lower_bounds[i] = i_lb
        upper_bounds[i] = i_ub

    return lower_bounds, upper_bounds


def get_range(star, neuron):
    u = LP.optimize_star(star.V[neuron, :], star.P_coef, star.P_ub, mode='max')
    l = LP.optimize_star(star.V[neuron, :], star.P_coef, star.P_ub, mode='min')

    print(u)
    print(l)
    print(star.center[neuron])
    lb = star.center[neuron] + l
    ub = star.center[neuron] + u

    return lb, ub


def to_starset(l_bs, u_bs):
    """
    :param l_bs: numpy array of Lower bounds
    :param u_bs: numpy array of corresponding upper bounds
    :return: A star set covering the region specified by the bounds
    """

    num_neurons = len(l_bs)
    center = (l_bs + u_bs) / 2
    width = (u_bs - l_bs) / 2

    V = np.identity(num_neurons)

    P_coef_ubs = np.identity(num_neurons)
    P_coef_lbs = -1 * np.identity(num_neurons)

    P_coef = np.vstack((P_coef_ubs, P_coef_lbs))

    P_ub = np.concatenate((width, width))

    star = StarSet(center, V, P_coef, P_ub)
    return star
