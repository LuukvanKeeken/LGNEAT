from tensorneat.src.tensorneat.pipeline import Pipeline
from tensorneat.src.tensorneat.algorithm.neat import NEAT
from tensorneat.src.tensorneat.algorithm.hyperneat import HyperNEATFeedForward, MLPSubstrate
from tensorneat.src.tensorneat.algorithm.hyperneat.hyperneat_feedforward_cust import HyperNEATFeedForwardCust
from tensorneat.src.tensorneat.genome import DefaultGenome
from tensorneat.src.tensorneat.common import ACT
import jax.numpy as jnp

from tensorneat.src.tensorneat.problem.func_fit import XOR3d

import time

start_time = time.time()

if __name__ == "__main__":

    algorithm=HyperNEATFeedForwardCust(
            substrate=MLPSubstrate(
                layers=[4, 50, 50, 1],
            ),
            neat=NEAT(
                pop_size=1000,
                species_size=20,
                survival_threshold=0.01,
                genome=DefaultGenome(
                    num_inputs=4,  # size of query coors
                    num_outputs=1,
                    init_hidden_layers=(),
                    output_transform=ACT.tanh,
                ),
            ),
            activation=ACT.tanh,
            output_transform=ACT.sigmoid,
        )
    
    pipeline = Pipeline(
        algorithm=algorithm,
        problem=XOR3d(),
        generation_limit=10000
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)

    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Approximate time/generation: {(time.time() - start_time)/pipeline.generation_limit:.2f} seconds\n")


    start_time_test = time.time()
    # show result
    for i in range(20):
        pipeline.show(state, best)
    print(f"Testing time: {time.time() - start_time_test} seconds\n")
    print(f"Average time per test: {(time.time() - start_time_test)/20} seconds\n")

    # visualize the best individual
    network = algorithm.neat.genome.network_dict(state, *best)
    print(algorithm.neat.genome.repr(state, *best))
    algorithm.neat.genome.visualize(network, save_path="./imgs/xor_CPPN_network_mlp.svg")

    transformed = algorithm.transform(state, best)
    seqs, h_nodes, h_conns, u_conns = transformed
    hyper_network = algorithm.hyper_genome.network_dict(state, h_nodes, h_conns)
    print(algorithm.hyper_genome.repr(state, h_nodes, h_conns))
    algorithm.hyper_genome.visualize(hyper_network, save_path="./imgs/xor_hyperneat_network_mlp.svg")

    