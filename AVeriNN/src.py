#!/usr/bin/env python3
import sys
from Onnx_to_NN import readOnnx_to_NN
from vnnlib import vnnlib_main
from starset import to_starset, reach_star
from NN import read_model_to_NN, reduce_NN, make_partition_clink
import numpy as np
from verification import verify_output

onnx_file = sys.argv[2]
vnnlib_file = sys.argv[3]
timeout = sys.argv[4]
results_file = sys.argv[5]

delta = 0.0

# Convert FFNN .onnx models to weights,bias and layer matrices
weights, bias, layers = readOnnx_to_NN(onnx_file)

# Parse vnnlib files
vnn_spec = vnnlib_main(onnx_file, vnnlib_file)

# Split the parsed vnnlib specification into input constraints and output constraints
inp_bounds = vnn_spec[0][0]
out_spec = vnn_spec[0][1]


# Collect bounds of variables from input constraints
lower_bounds, upper_bounds = list(zip(*inp_bounds))

# Convert input box bounds to star polyhedron
inp_star = to_starset(np.array(lower_bounds), np.array(upper_bounds))

# Convert weights,bias,layer data matrices into a NN object
neural_network = read_model_to_NN(weights, bias, layers)
# Compute a partition for the above NN object for a fixed delta
Partition = make_partition_clink(neural_network, delta)
# Reduce the above NN object w.r.t to the computed partition
reduced_network = reduce_NN(neural_network, Partition)

stars, splits, num_stars = reach_star(reduced_network, [inp_star], method='feasibility')

result = verify_output(stars, out_spec)

with open(results_file, 'w') as f:
    f.write(result)

