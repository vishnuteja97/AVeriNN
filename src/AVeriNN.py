#!/usr/bin/env python3
import sys
import time
from Onnx_to_NN import readOnnx_to_NN
from vnnlib import vnnlib_main
from Istarset import reach_star, to_starset
from IVerification import verify_output, get_range_star
from INN import *
import numpy as np
import matlab_interface

t1 = time.time()
onnx_file = sys.argv[1]
vnnlib_file = sys.argv[2]

# Convert FFNN .onnx models to weights,bias and layer matrices
weights, bias, layers = readOnnx_to_NN(onnx_file)

# delta is a vector of parameters. Num of params must be equal to number of hidden layers
delta = [500, 500, 500, 500, 500, 500]

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
neural_network = read_model_to_INN(weights, bias, layers)

# Compute a partition for the above NN object for a fixed delta
Partition = make_partition_clink(neural_network, delta)
# Reduce the above NN object w.r.t to the computed partition
reduced_network = reduce_INN(neural_network, Partition)
print(f"The architecture of the reduced network is:\n{reduced_network.get_features()}")


stars, num_stars = reach_star(reduced_network, [inp_star], method='feasibility')

print(verify_output(stars, out_spec))
L, U = get_range_star(stars)
print(f'[{L[0]}, {U[0]}]')
print(f'[{L[1]}, {U[1]}]')
print(f'[{L[2]}, {U[2]}]')
print(f'[{L[3]}, {U[3]}]')
print(f'[{L[4]}, {U[4]}]')

print(num_stars)
matlab_interface.ENG.quit()
t2 = time.time()

print(t2-t1)
