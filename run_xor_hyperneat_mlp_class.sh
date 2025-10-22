#!/bin/bash

#script to be run server-side

# install requirements, then execute script with given arguments

cd /project_antwerp/code/LGNEAT
# pip install wfdb
python -m pip install -U pip

pip install -U "jax[cuda12]"



#for 1 run:
python3 "xor_hyperneat_mlp_class.py" 