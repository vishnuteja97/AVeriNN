#!/bin/bash
# 2 arguments, first is path to the .onnx file, second is path to .vnnlib file 
# Author: Vishnu B

VERSION_STRING=v1

if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

TOOL_NAME=AVeriNN
ONNX_FILE=$1
VNNLIB_FILE=$2

echo "Running '$TOOL_NAME' on benchmark instance with onnx file '$ONNX_FILE' with vnnlib file '$VNNLIB_FILE'."

DIR=$(dirname $(dirname $(realpath $0)))
export PYTHONPATH="$PYTHONPATH:$DIR"
export PYTHONPATH="$PYTHONPATH:$DIR/src"

python3 -m src.AVeriNN "$ONNX_FILE" "$VNNLIB_FILE" 
