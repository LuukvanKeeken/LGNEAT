#!/bin/bash

class="$1"
add_node_rate="$2"
add_conn_rate="$3"
start_seed="$4"
num_elites="$5"

#script to be run server-side

# install requirements, then execute script with given arguments

cd /project_antwerp/code/LGNEAT
# pip install wfdb
python -m pip install -U pip

pip install -U "jax[cuda12]"



if [ "$#" -eq 1 ]; then
    python3 "hyperneat_lgn_mnist_1vsrest.py" --single_class="$class"
elif [ "$#" -eq 2 ]; then
    # Example: use class and add_node_rate
    python3 "hyperneat_lgn_mnist_1vsrest.py" --single_class="$class" --add_node_rate="$add_node_rate"
elif [ "$#" -eq 3 ]; then
    # Example: use class, add_node_rate, add_conn_rate
    python3 "hyperneat_lgn_mnist_1vsrest.py" --single_class="$class" --add_node_rate="$add_node_rate" --add_conn_rate="$add_conn_rate"
elif [ "$#" -eq 4 ]; then
    python3 "hyperneat_lgn_mnist_1vsrest.py" --single_class="$class" --add_node_rate="$add_node_rate" --add_conn_rate="$add_conn_rate" --start_seed="$start_seed"
elif [ "$#" -eq 5 ]; then
    python3 "hyperneat_lgn_mnist_1vsrest.py" --single_class="$class" --add_node_rate="$add_node_rate" --add_conn_rate="$add_conn_rate" --start_seed="$start_seed" --num_elites="$num_elites"
else
    python3 "hyperneat_lgn_mnist_1vsrest.py" 
fi