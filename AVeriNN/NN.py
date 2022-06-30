import numpy as np
import onnx
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd


class Node:
    def __init__(self, l):
        self.layer_num = l
        self.act_func = 0
        self.bias = 0

    def update_bias(self, b):
        self.bias = b

    def update_act_func(self, af):
        self.act_func = af

    def update_layer_num(self, l):
        self.layer_num = l


class NN:
    def __init__(self):
        self.layers = []
        self.graph = []

    def compute_pre_sum(self, curr_node, prev_node_indices):
        sum = 0.0

        curr_layer = curr_node.layer_num
        for index in prev_node_indices:
            prev_node = self.layers[curr_layer - 1][index]
            sum = sum + self.graph[curr_layer - 1][curr_node][prev_node]

        return sum
        '''

        return sum(
            [self.graph[curr_node.layer_num - 1][curr_node][self.layers[curr_node.layer_num - 1][index]] for index in
             prev_node_indices])
        '''

    def compute_pre_sum_vector(self, curr_node, p_i):
        """Given a partition 'p_i' of previous layer, return the presum vector(bias included) for the curr_node"""

        return np.array([self.compute_pre_sum(curr_node, p) for p in p_i] + [curr_node.bias])

    def get_features(self):
        """Returns the architecture of NN as a list of number of nodes in each layer"""
        L = []
        for i in range(len(self.layers)):
            L.append(len(self.layers[i]))

        return L

    def get_weights(self):
        weights = []
        for layer_num in range(1, len(self.layers)):
            curr_layer = self.layers[layer_num]
            prev_layer = self.layers[layer_num - 1]

            layer_weights = []
            for node_out in curr_layer:
                row_weights = []
                for node_in in prev_layer:
                    row_weights.append(self.graph[layer_num - 1][node_out][node_in])
                layer_weights.append(row_weights)

            weights.append(np.array(layer_weights))

        return weights

    def get_bias(self):
        bias = []
        for layer_num in range(1, len(self.layers)):
            curr_layer = self.layers[layer_num]

            layer_bias = [node.bias for node in curr_layer]
            bias.append(np.array(layer_bias))

        return bias


def create_layer_nodes(num_of_nodes, layer_num):
    list_of_nodes = [Node(layer_num) for i in range(num_of_nodes)]

    return list_of_nodes


def make_partition(nn, delta):
    """Naive method for partitioning"""

    P = [[[i] for i in range(len(nn.layers[0]))]]
    num_layers = len(nn.layers)

    for i in range(1, num_layers - 1):
        presum_memory = {ind: nn.compute_pre_sum_vector(nn.layers[i][ind], P[i - 1]) for ind in
                         range(len(nn.layers[i]))}
        P_i = [[0]]

        for index in range(1, len(nn.layers[i])):
            j = 0
            while j < len(P_i):
                part_node_index = P_i[j][0]

                if (np.absolute(presum_memory[index] - presum_memory[part_node_index]) <= delta / 2).all():
                    P_i[j].append(index)
                    break
                j = j + 1
            if j == len(P_i):
                P_i.append([index])

        P.append(P_i)

    P.append([[i] for i in range(len(nn.layers[-1]))])
    return P


def make_partition_clink(nn, delta):
    """Perform hierarchial clustering for each layer to produce partition"""

    P = [[[i] for i in range(len(nn.layers[0]))]]
    num_layers = len(nn.layers)

    for i in range(1, num_layers - 1):
        presums = [nn.compute_pre_sum_vector(nn.layers[i][ind], P[i - 1]) for ind in range(len(nn.layers[i]))]

        Z = linkage(presums, 'single', 'euclidean')
        clusters = fcluster(Z, t=delta, criterion='distance')

        P_i = pd.Series(range(len(clusters))).groupby(clusters, sort=False).apply(list).tolist()

        P.append(P_i)

    P.append([[i] for i in range(len(nn.layers[-1]))])
    return P


def reduce_NN(N_in, P):
    red_NN = NN()
    num_layers = len(P)

    for layer in range(num_layers):
        red_NN.layers.append(create_layer_nodes(len(P[layer]), layer))

        for k in range(len(P[layer])):
            bias_node_index = P[layer][k][0]
            bias = N_in.layers[layer][bias_node_index].bias

            red_NN.layers[-1][k].update_bias(bias)

    for layer in range(1, num_layers):
        red_NN.graph.append({})
        for k in range(len(P[layer])):
            curr_node_ori = N_in.layers[layer][P[layer][k][0]]
            curr_node = red_NN.layers[layer][k]

            layer_dict = {}
            for m in range(len(P[layer - 1])):
                prev_node = red_NN.layers[layer - 1][m]
                presum_weight = N_in.compute_pre_sum(curr_node_ori, P[layer - 1][m])
                layer_dict.update({prev_node: presum_weight})

            red_NN.graph[layer - 1].update({curr_node: layer_dict})

    return red_NN


def read_model_to_NN(file_weights, file_bias, file_layers):
    outputNN = NN()

    def update_nodes_info(layer_num):
        curr_biases = file_bias[layer_num - 1]
        curr_layer_nodes = outputNN.layers[layer_num]

        for k in range(len(curr_biases)):
            curr_layer_nodes[k].update_bias(curr_biases[k])

    def read_weights_to_graph(layer_num):
        curr_layer_weights = file_weights[layer_num - 1]
        graph_dic = {}
        shape_weights = file_weights[layer_num - 1].shape

        for r in range(shape_weights[0]):
            curr_node = outputNN.layers[layer_num][r]
            graph_dic.update({curr_node: dict(zip(outputNN.layers[layer_num - 1], curr_layer_weights[r, :]))})

        outputNN.graph.append(graph_dic)

    num_layers = len(file_layers)

    outputNN.layers.append(create_layer_nodes(file_layers[0], 0))  # initialising starting layer of nodes

    for j in range(1, num_layers):
        outputNN.layers.append(create_layer_nodes(file_layers[j], j))
        update_nodes_info(j)

        read_weights_to_graph(j)

    return outputNN


'''
wei, bia, lay = readfile('ACASXU_experimental_v2a_1_6.nnet')
init_NN = read_model_to_NN(wei, bia, lay)


model = onnx.load('C:/Users/Vishnu Teja B/Desktop/Python/NNBS/ACAS_Xu_BM/ACASXU_run2a_1_1_batch_2000.onnx')

print(model
      )
'''
