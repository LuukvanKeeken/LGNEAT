from tensorneat.src.tensorneat.pipeline import Pipeline
from tensorneat.src.tensorneat.algorithm.neat import NEAT
from tensorneat.src.tensorneat.algorithm.hyperneat import HyperNEATFeedForward, MLPSubstrate
from tensorneat.src.tensorneat.algorithm.hyperneat.hyperneat_feedforward_cust import HyperNEATFeedForwardCust
from tensorneat.src.tensorneat.genome import DefaultGenome
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
if not os.path.exists(f"results/{timestamp}_mlp"):
    os.makedirs(f"results/{timestamp}_mlp")

# Create a directory in results/timestamp for images
if not os.path.exists(f"results/{timestamp}_mlp/imgs"):
    os.makedirs(f"results/{timestamp}_mlp/imgs")

if __name__ == "__main__":

    layers = [7, 10, 10, 2]

    neat_inputs = 4
    neat_outputs = 1
    neat_hidden_layers = (2,)

    algorithm=HyperNEATFeedForwardCust(
            substrate=MLPSubstrate(
                layers=layers,
            ),
            neat=NEAT(
                pop_size=10,
                species_size=20,
                survival_threshold=0.01,
                genome=DefaultGenome(
                    num_inputs=neat_inputs,  # size of query coors
                    num_outputs=neat_outputs,
                    init_hidden_layers=neat_hidden_layers,
                    output_transform=ACT.tanh,
                ),
            ),
            activation=ACT.tanh,
            output_transform=ACT.identity,
        )
    
    # Check if input coordinates are inside a circle
    # # Return [1, 0] if inside, else [0, 1]
    # def inside_circle(inputs, radius=0.5):
    #     x, y = inputs
    #     res = jnp.square(x) + jnp.square(y)
        
    #     return jnp.where(res <= radius**2, jnp.array([1]), jnp.array([0]))
    
    # inside_circle_problem = CustomFuncFit(
    #     func = inside_circle,
    #     low_bounds = [-1, -1],
    #     upper_bounds = [1, 1],
    #     method = "sample",
    #     num_samples = 20
    # )

    # pipeline = Pipeline(
    #     algorithm=algorithm,
    #     problem=inside_circle_problem,
    #     generation_limit=10000,
    #     seed=3
    # )

    # Inputs is a vector of ones and zeros. Count the number
    # of ones, and return 1 if that number is even, otherwise 0
    def even_ones(inputs):
        count_ones = jnp.sum(inputs)
        return jnp.where(count_ones % 2 == 0, jnp.array([1, 0]), jnp.array([0, 1]))


    even_problem = CustomFuncFit(
        func = even_ones,
        low_bounds = jnp.zeros(6),
        upper_bounds = jnp.ones(6)*1.1,
        method = "grid",
        step_size = jnp.ones(6)
    )

    pipeline = Pipeline(
        algorithm=algorithm,
        problem=even_problem,
        fitness_target=-0.1,
        generation_limit=20000,
        seed=4
    )


    with open(f"results/{timestamp}_mlp/settings.txt", "w") as f_settings:
        f_settings.write(f"Generations: {pipeline.generation_limit}\n")
        f_settings.write(f"Fitness target: {pipeline.fitness_target}\n")
        f_settings.write(f"Population size: {algorithm.neat.pop_size}\n")
        f_settings.write(f"Substrate layers: {layers}\n")
        f_settings.write(f"Species: {algorithm.neat.species_controller.species_size}\n")
        f_settings.write(f"NEAT init. shape {neat_inputs, neat_hidden_layers, neat_outputs}\n")
        f_settings.write(f"Seed: {pipeline.seed}\n")
        f_settings.write(f"Problem task: {pipeline.problem.__class__.__name__}\n")
        f_settings.write(f"Elites: {algorithm.neat.species_controller.species_elitism}\n")

    print("Starting training ...")
    with open(f"results/{timestamp}_mlp/log.txt", "w") as f_log:
        with contextlib.redirect_stdout(f_log):
            # initialize state
            state = pipeline.setup()
            # print(state)
            # run until terminate
            filename = f"results/{timestamp}_mlp/current_gen.txt"
            state, best = pipeline.auto_run(state, filename)

            best_transformed = algorithm.transform(state, best)
            print("Final evaluation on the whole dataset:")
            print(even_problem.evaluate(state, None, algorithm.forward, best_transformed))

    print("Finished training.")



    with open(f"results/{timestamp}_mlp/best.txt", "w") as f_best:
        with contextlib.redirect_stdout(f_best):
            print(f"Total time: {time.time() - start_time:.2f} seconds")
            print(f"Approximate time/generation: {(time.time() - start_time)/pipeline.generation_limit:.2f} seconds\n")


            start_time_test = time.time()
            # show result
            pipeline.show(state, best)
            print(f"Testing time: {time.time() - start_time_test} seconds\n")

            # visualize the best individual
            network = algorithm.neat.genome.network_dict(state, *best)
            print(algorithm.neat.genome.repr(state, *best))
            algorithm.neat.genome.visualize(network, save_path=f"results/{timestamp}_mlp/imgs/xor_CPPN_network_mlp.svg")

            transformed = algorithm.transform(state, best)
            seqs, h_nodes, h_conns, u_conns = transformed
            hyper_network = algorithm.hyper_genome.network_dict(state, h_nodes, h_conns)
            print(algorithm.hyper_genome.repr(state, h_nodes, h_conns))
            algorithm.hyper_genome.visualize(hyper_network, save_path=f"results/{timestamp}_mlp/imgs/xor_hyperneat_network_mlp.svg")

    