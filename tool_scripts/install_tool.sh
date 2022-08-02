#!/bin/bash
#Installation script
#Author: Vishnu B

TOOL_NAME=AVeriNN
VERSION_STRING=v1 

echo "Installing $TOOL_NAME"

DIR=$(dirname $(dirname $(realpath $0)))

apt-get update &&
apt-get install -y python3 python3-pip &&

#Dependencies
pip3 install numpy==1.22.3
pip3 install scipy==1.8.0
pip3 install pandas==1.4.2
pip3 install onnx==1.11.0
pip3 install onnxruntime==1.11.1
pip3 install protobuf==3.19.4
pip3 install pyinterval==1.2.0

#Setting up CVXOPT with access to GLPK.
apt-get install python-cvxopt
apt-get install libglpk-dev
CVXOPT_BUILD_GLPK=1 pip3 install cvxopt
