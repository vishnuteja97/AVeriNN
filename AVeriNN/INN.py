import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
from Interval import Imatrix, to_Imatrix
from interval import interval


class Node:
    def __init__(self, l):
        self.layer_num = l
        self.bias = interval(0)

    def update_bias(self, b):
        if isinstance(b, (int, float, np.float32, np.float64)):
            self.bias = interval(b)
        elif isinstance(b, interval):
            self.bias = b
        else:
            raise TypeError(f"Unexpected bias type, got {type(b)}")

    def update_layer_num(self, l):
        self.layer_num = l


class INN:
    def __init__(self):
        self.layers = []
        self.graph = []

    def get_nodes(self, layer_num, indices=None):
        """Returns a list of node objects of a specified layer and indices(optional)"""
        if indices is None:
            return self.layers[layer_num]
        else:
            return [self.layers[layer_num][index] for index in indices]

    def get_edge(self, layer_num, out_node, inp_node):
        """Note: NN graph is stored with reversed edges"""

        return self.graph[layer_num][out_node][inp_node]

    def get_features(self):
        """Returns the architecture of INN as a list of number of nodes in each layer"""
        L = [len(layer) for layer in self.layers]

        return L

    def get_weights(self):
        """Returns the list of weight matrices(of each layer)"""
        weights = []
        for layer_num in range(1, len(self.layers)):
            curr_layer = self.layers[layer_num]
            prev_layer = self.layers[layer_num - 1]

            layer_weights = []
            for node_out in curr_layer:
                row_weights = []
                for node_in in prev_layer:
                    row_weights.append(self.get_edge(layer_num - 1, node_out, node_in))
                layer_weights.append(row_weights)

            weights.append(to_Imatrix(layer_weights))

        return weights

    def get_bias(self):
        """Returns the list of bias vectors(of each layer)"""
        bias = []
        for layer_num in range(1, len(self.layers)):
            curr_layer = self.layers[layer_num]

            layer_bias = [[node.bias] for node in curr_layer]
            bias.append(to_Imatrix(layer_bias))

        return bias

    def compute_pre_sum(self, curr_node, prev_node_indices):
        """Gives the presum for a node w.r.t nodes in previous layer"""
        sum = interval(0.0)

        prev_layer_num = curr_node.layer_num - 1
        prev_nodes = self.get_nodes(prev_layer_num, prev_node_indices)

        for prev_node in prev_nodes:
            sum = sum + self.get_edge(prev_layer_num, curr_node, prev_node)

        return sum

    def compute_pre_sum_vector(self, curr_node, p_i):
        """Given a partition 'p_i' of previous layer, return the presum vector(bias included) for the curr_node
           The pre_sum vector is an interval vector, which is converted to a normal vector by separating endpoints
        """

        int_pre_sum = [self.compute_pre_sum(curr_node, p) for p in p_i] + [curr_node.bias]
        lower_pre_sum = [x[0].inf for x in int_pre_sum]
        upper_pre_sum = [x[0].sup for x in int_pre_sum]

        return np.array(lower_pre_sum + upper_pre_sum)

    def compute_int_hull(self,  prev_node_indices, curr_node_indices, curr_layer):
        """Given a set of indices of nodes in previous layer and nodes in current layer
        compute the interval hull of all the weights.
        """
        size = len(prev_node_indices)
        curr_nodes = self.get_nodes(curr_layer, curr_node_indices)
        prev_nodes = self.get_nodes(curr_layer-1, prev_node_indices)

        intervals = tuple([self.get_edge(curr_layer-1, curr_node, prev_node) for curr_node in curr_nodes for prev_node in prev_nodes])

        return size * interval.hull(intervals)


def create_layer_nodes(num_of_nodes, layer_num):
    list_of_nodes = [Node(layer_num) for _ in range(num_of_nodes)]

    return list_of_nodes


def read_model_to_INN(file_weights, file_bias, file_layers):
    """Create an INN object using weight matrices(type: list of n-d arrays) and biases"""
    outputINN = INN()

    def update_nodes_info(layer_num):
        curr_biases = file_bias[layer_num - 1]
        curr_layer_nodes = outputINN.layers[layer_num]

        for k in range(len(curr_biases)):
            curr_layer_nodes[k].update_bias(curr_biases[k])

    def read_weights_to_graph(layer_num):
        curr_layer_weights = file_weights[layer_num - 1]
        graph_dic = {}
        shape_weights = file_weights[layer_num - 1].shape

        for r in range(shape_weights[0]):
            curr_node = outputINN.layers[layer_num][r]
            int_weights = [interval(w) for w in curr_layer_weights[r, :]]

            graph_dic.update({curr_node: dict(zip(outputINN.get_nodes(layer_num - 1), int_weights))})

        outputINN.graph.append(graph_dic)

    num_layers = len(file_layers)

    outputINN.layers.append(create_layer_nodes(file_layers[0], 0))  # initialising starting layer of nodes

    for j in range(1, num_layers):
        outputINN.layers.append(create_layer_nodes(file_layers[j], j))
        update_nodes_info(j)

        read_weights_to_graph(j)

    return outputINN


def make_partition_clink(inn, delta):
    """Perform clustering for each layer to produce partition, delta: 1-d array"""

    P = [[[i] for i in range(len(inn.layers[0]))]]
    num_layers = len(inn.layers)

    for i in range(1, num_layers - 1):
        presums = [inn.compute_pre_sum_vector(inn.layers[i][ind], P[i - 1]) for ind in range(len(inn.layers[i]))]

        Z = linkage(np.array(presums), 'single', 'euclidean')
        clusters = fcluster(Z, t=delta[i-1], criterion='distance')

        P_i = pd.Series(range(len(clusters))).groupby(clusters, sort=False).apply(list).tolist()

        P.append(P_i)

    P.append([[i] for i in range(len(inn.layers[-1]))])  # P_n 'last layer'
    return P


def reduce_INN(N_in, P):
    red_NN = INN()
    num_layers = len(P)

    # Create reduced INN and update biases of all nodes.
    for layer in range(num_layers):
        red_NN.layers.append(create_layer_nodes(len(P[layer]), layer))

        for k in range(len(P[layer])):
            bias_node_indices = P[layer][k]
            biases = tuple([N_in.layers[layer][index].bias for index in bias_node_indices])
            bias = interval.hull(biases)

            red_NN.layers[-1][k].update_bias(bias)

    # Update the weights of all nodes
    for layer in range(1, num_layers):
        red_NN.graph.append({})
        for k in range(len(P[layer])):
            curr_node = red_NN.layers[layer][k]

            layer_dict = {}
            for m in range(len(P[layer - 1])):
                prev_node = red_NN.layers[layer - 1][m]

                interval_hull = N_in.compute_int_hull(P[layer-1][m], P[layer][k], layer)

                layer_dict.update({prev_node: interval_hull})

            red_NN.graph[layer - 1].update({curr_node: layer_dict})

    return red_NN

