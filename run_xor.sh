#!/bin/bash

#script to be run server-side

# install requirements, then execute script with given arguments

cd /project_antwerp/code/LGNEAT
# pip install wfdb
python -m pip install -U pip

pip install -U "jax[cuda12]"
pip install git+https://github.com/EMI-Group/tensorneat.git






#for 1 run:
python3 "jaxtest.py" 