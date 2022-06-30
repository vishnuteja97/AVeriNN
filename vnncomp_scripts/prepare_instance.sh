TOOL_NAME=AVeriNN
VERSION_STRING=v1

if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '${VERSION_STRING}', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4

echo "Preparing ${TOOL_NAME} for benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE' and vnnlib file '$VNNLIB_FILE'"

if [ "$2" != "cartpole" ] && [ "$2" != "dubins_rejoin_task" ] && [ "$2" != "lunarlander" ] && [ "$2" != "mnist_fc" ] && [ "$2" != "tllverifybench" ]
then
	echo "This category '$2' is not supported"
	exit 1 
fi

killall -q python3

echo "Instance Prepared"
exit 0
	
