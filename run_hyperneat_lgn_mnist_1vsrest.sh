#!/bin/bash

class="$1"

#script to be run server-side

# install requirements, then execute script with given arguments

cd /project_antwerp/code/LGNEAT
# pip install wfdb
python -m pip install -U pip

pip install -U "jax[cuda12]"



#for 1 run:
python3 "hyperneat_lgn_mnist_1vsrest.py" --single_class="$class"