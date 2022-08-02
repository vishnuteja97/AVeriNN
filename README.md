The tool is mostly built on python but uses a single matlab script for interval matrix multiplications.

Instructions for installation:

Python-setup
1.) Navigate to the 'scripts' directory.
2.) Execute the 'install_tool.sh' script. This will automatically install all python dependencies.

MATLAB-setup
1.) Download and install MATLAB-R2022A or higher.
2.) Download CORA toolbox 2021 from https://tumcps.github.io/CORA/
3.) Add the folder containing 'intMatMul.m' i.e the 'src' folder to the MATLAB search path.
4.) Navigate to the folder where MATLAB is installed.
5.) Go to /R2022a/extern/engines/python/
6.) Run the setup.py to install python api for MATLAB engine.
