#!/bin/bash
#Installation script for VNNCOMP22
#Author: Vishnu B

TOOL_NAME=AVeriNN
VERSION_STRING=v1 

echo "Installing $TOOL_NAME"

DIR=$(dirname $(dirname $(realpath $0)))

apt-get update &&
apt-get install -y python3 python3-pip &&
apt-get install -y psmic && # For killall used in prepare_instance.sh

pip3 install -r "$DIR/requirements.txt"

#Setting up CVXOPT with access to GLPK.
apt-get install python-cvxopt
apt-get install libglpk-dev
CVXOPT_BUILD_GLPK=1 pip3 install cvxopt
