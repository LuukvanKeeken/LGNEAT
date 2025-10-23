from tensorneat.src.tensorneat.pipeline import Pipeline
from tensorneat.src.tensorneat.algorithm.neat import NEAT
from tensorneat.src.tensorneat.algorithm.hyperneat import HyperNEATFeedForward, MLPSubstrateLEO
from tensorneat.src.tensorneat.algorithm.hyperneat.hyperneat_feedforward_cust_two_inputs import HyperNEATFeedForwardCustTwoInputs
from tensorneat.src.tensorneat.genome import DefaultGenomeCPPN
from tensorneat.src.tensorneat.common import ACT
import jax.numpy as jnp
from tensorneat.src.tensorneat.problem.func_fit import CustomFuncFit

import time
import os
import contextlib

start_time = time.time()

# Check if results directory exists, if not, create it
if not os.path.exists("results"):
    os.makedirs("results")

# Create directory in results with current timestamp,
# if it does not exist
timestamp = time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(f"results/{timestamp}_mlp_two_inputs"):
    os.makedirs(f"results/{timestamp}_mlp_two_inputs")

# Create a directory in results/timestamp for images
if not os.path.exists(f"results/{timestamp}_mlp_two_inputs/imgs"):
    os.makedirs(f"results/{timestamp}_mlp_two_inputs/imgs")

if __name__ == "__main__":

    layers = [3, 10, 10, 1]
    neat_inputs = 4
    neat_outputs = 2
    neat_hidden_layers = ()

    algorithm=HyperNEATFeedForwardCustTwoInputs(
            substrate=MLPSubstrateLEO(
                layers=layers,
            ),
            neat=NEAT(
                pop_size=1000,
                species_size=20,
                survival_threshold=0.01,
                genome=DefaultGenomeCPPN(
                    num_inputs=neat_inputs,  # size of query coors
                    num_outputs=neat_outputs,
                    init_hidden_layers=neat_hidden_layers,
                    output_transform=ACT.sigmoid,
                ),
            ),
            activation=ACT.tanh,
            output_transform=ACT.sigmoid,
        )
    
    # Check if input coordinates are inside a circle
    # Return [1, 0] if inside, else [0, 1]
    def inside_circle(inputs, radius=0.5):
        x, y = inputs
        res = jnp.square(x) + jnp.square(y)
        
        return jnp.where(res <= radius**2, jnp.array([1]), jnp.array([0]))
    
    inside_circle_problem = CustomFuncFit(
        func = inside_circle,
        low_bounds = [-1, -1],
        upper_bounds = [1, 1],
        method = "sample",
        num_samples = 20
    )

    pipeline = Pipeline(
        algorithm=algorithm,
        problem=inside_circle_problem,
        generation_limit=10000,
        seed=3
    )

    with open(f"results/{timestamp}_mlp_two_inputs/settings.txt", "w") as f_settings:
        f_settings.write(f"Generations: {pipeline.generation_limit}\n")
        f_settings.write(f"Fitness target: {pipeline.fitness_target}\n")
        f_settings.write(f"Population size: {algorithm.neat.pop_size}\n")
        f_settings.write(f"Substrate layers: {layers}\n")
        f_settings.write(f"Species: {algorithm.neat.species_controller.species_size}\n")
        f_settings.write(f"NEAT init. shape {neat_inputs, neat_hidden_layers, neat_outputs}\n")
        f_settings.write(f"Seed: {pipeline.seed}\n")
        f_settings.write(f"Problem task: {pipeline.problem.__class__.__name__}\n")

    print("Starting training ...")
    with open(f"results/{timestamp}_mlp_two_inputs/log.txt", "w") as f_log:
        with contextlib.redirect_stdout(f_log):

            # initialize state
            state = pipeline.setup()
            # print(state)
            # run until terminate
            filename = f"results/{timestamp}_mlp_two_inputs/current_gen.txt"
            state, best = pipeline.auto_run(state, filename)

            print(f"Total time: {time.time() - start_time:.2f} seconds")
            print(f"Approximate time/generation: {(time.time() - start_time)/pipeline.generation_limit:.2f} seconds\n")

    print(f"Finished training")

    with open(f"results/{timestamp}_mlp_two_inputs/best.txt", "w") as f_best:
        with contextlib.redirect_stdout(f_best):
            start_time_test = time.time()
            
            pipeline.show(state, best)
            print(f"Testing time: {time.time() - start_time_test} seconds\n")

            # visualize the best individual
            network = algorithm.neat.genome.network_dict(state, *best)
            print(algorithm.neat.genome.repr(state, *best))
            algorithm.neat.genome.visualize(network, save_path=f"results/{timestamp}_mlp_two_inputs/imgs/neat_CPPN_network.svg")

            transformed = algorithm.transform(state, best)
            seqs, h_nodes, h_conns, u_conns = transformed
            hyper_network = algorithm.hyper_genome.network_dict(state, h_nodes, h_conns)
            print(algorithm.hyper_genome.repr(state, h_nodes, h_conns))
            algorithm.hyper_genome.visualize(hyper_network, save_path=f"results/{timestamp}_mlp_two_inputs/imgs/hyperneat_network.svg")

    