from INN import read_model_to_INN
from Istarset import reach_star, to_starset
import numpy as np
from pyexcel_ods import save_data
from collections import OrderedDict


def create_INN(wt_range, hid_lay, num_nodes):
    wt_range_size = wt_range[1] - wt_range[0]
    layers = [num_nodes] * (hid_lay + 2)
    wts = [(wt_range_size * np.random.rand(num_nodes, num_nodes)) + wt_range[0] for _ in range(hid_lay + 1)]
    bias = [(wt_range_size * np.random.rand(num_nodes)) + wt_range[0] for _ in range(hid_lay + 1)]

    return read_model_to_INN(wts, bias, layers)

def create_INN2(hid_lay, num_nodes, pos_perc):
    layers = [num_nodes] * (hid_lay + 2)
    wts = []
    bias = []
    for _ in range(hid_lay+1):
        new_wt = 2.0*(np.random.rand(num_nodes, num_nodes) > pos_perc) - 1.0
        new_bias = 2.0 * (np.random.rand(num_nodes) > pos_perc) - 1.0
        wts.append(new_wt)
        bias.append(new_bias)

    return read_model_to_INN(wts, bias, layers)


def experiments(wt_range, hid_lay, num_nodes, pos_perc):
    wt_range_size = wt_range[1] - wt_range[0]

    record_stars = 0
    record_t = 0

    for _ in range(10):
        inn = create_INN2(hid_lay, num_nodes, pos_perc)
        inp_lb = (wt_range_size * np.random.rand(num_nodes)) + wt_range[0]
        inp_ub = (wt_range_size * np.random.rand(num_nodes)) + wt_range[0]

        inp_star = to_starset(inp_lb, inp_ub)
        stars, num_stars, t = reach_star(inn, [inp_star], method='feasibility')

        record_t += np.array(t)
        record_stars += np.array(num_stars)

    return record_t / 5, record_stars / 5


hid_lay = 6
wt_range = (-1.0, 1.0)
data = OrderedDict()
rows = [['Nodes', 'Layers', 'pos_perc', 'time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'out_stars1', 'out_stars2', 'out_stars3', 'out_stars4', 'out_stars5', 'out_stars6', 'out_stars7'], []]

for nodes in range(2, 7):
    for pperc in [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]:
        s, t = experiments(wt_range, hid_lay, nodes, pperc)

        rows.append([nodes, hid_lay, str(pperc), str(s[0]), str(s[1]), str(s[2]), str(s[3]), str(s[4]), str(s[5]),  str(t[0]), str(t[1]), str(t[2]), str(t[3]), str(t[4]), str(t[5])])

data.update({"Sheet 1": rows})
save_data("experiments5.ods", data)
