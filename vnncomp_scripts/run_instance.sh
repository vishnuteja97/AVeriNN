#!/bin/bash
# 6 arguments, first is "v1", second is a benchmark category identifier(For ex: "cartpole"), third is path to the .onnx file, fourth is path to .vnnlib file, fifth is a path to the results file, and sixth is a timeout in seconds.
# Author: Vishnu B

VERSION_STRING=v1

if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

TOOL_NAME=AVeriNN
CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo "Running '$TOOL_NAME' on benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE' with vnnlib file '$VNNLIB_FILE' with results file '$RESULTS_FILE' and with timeout '$TIMEOUT' in seconds."

DIR=$(dirname $(dirname $(realpath $0)))
export PYTHONPATH="$PYTHONPATH:$DIR/src"

python3 -m AVeriNN.src "$ONNX_FILE" "$VNNLIB_FILE" "$TIMEOUT" "$RESULTS_FILE"
