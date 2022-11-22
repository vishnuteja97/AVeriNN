import sys
from Onnx_to_NN import readOnnx_to_NN
from vnnlib import vnnlib_main
import numpy as np
from INN import *
from Istarset import *
from IVerification import *

onnx_file = sys.argv[1]
vnnlib_file = sys.argv[2]
delta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

c_in = np.array([0.6000000238418579, 0.013908768072724342, -0.19755497574806213, 0.44999998807907104, -0.5])
c_out = np.array([-0.01724148727953434, -0.018129119649529457, -0.018222741782665253, -0.017442017793655396, -0.01837892271578312])
# Convert FFNN .onnx models to weights,bias and layer matrices
weights, bias, layers = readOnnx_to_NN(onnx_file)

num_layers = len(weights)
counter_example = [c_in.reshape((-1, 1))]

for i in range(num_layers - 1):
    new_c = np.dot(weights[i], counter_example[-1])
    new_c += bias[i].reshape(-1, 1)

    for j in range(len(new_c)):
        if new_c[j][0] < 0:
            new_c[j][0] = 0

    counter_example.append(new_c)

new_c = np.dot(weights[-1], counter_example[-1]) + bias[-1].reshape(-1, 1)
counter_example.append(new_c)


vnn_spec = vnnlib_main(onnx_file, vnnlib_file)

# Split the parsed vnnlib specification into input constraints and output constraints
inp_bounds = vnn_spec[0][0]
out_spec = vnn_spec[0][1]


# Collect bounds of variables from input constraints
lower_bounds, upper_bounds = list(zip(*inp_bounds))
print(f'counterexample: {counter_example[0]}')
print(f'lower bounds : {lower_bounds}')
print(f'upper_bounds : {upper_bounds}')
# Convert input box bounds to star polyhedron
inp_star = to_starset(np.array(lower_bounds), np.array(upper_bounds), counter_example[0])

check_star_contains_point(inp_star, counter_example[0])

# Convert weights,bias,layer data matrices into a NN object
neural_network = read_model_to_INN(weights, bias, layers)

# Compute a partition for the above NN object for a fixed delta
Partition = make_partition_clink(neural_network, delta)
# Reduce the above NN object w.r.t to the computed partition
reduced_network = reduce_INN(neural_network, Partition)
print(f"The architecture of the reduced network is:\n{reduced_network.get_features()}")

stars, num_stars = reach_star(reduced_network, [inp_star], method='approx')

print(verify_output(stars, out_spec))



