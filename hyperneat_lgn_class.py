from tensorneat.src.tensorneat.pipeline import Pipeline
from tensorneat.src.tensorneat.algorithm.neat import NEAT
from tensorneat.src.tensorneat.algorithm.hyperneat import LGNSubstrateLEO
from tensorneat.src.tensorneat.algorithm.hyperneat import HyperNEATFeedForwardLGNClass
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
if not os.path.exists(f"results/{timestamp}_lgn"):
    os.makedirs(f"results/{timestamp}_lgn")

# Create a directory in results/timestamp for images
if not os.path.exists(f"results/{timestamp}_lgn/imgs"):
    os.makedirs(f"results/{timestamp}_lgn/imgs")

if __name__ == "__main__":

    layers = [6, 10, 10, 10, 10, 10]

    neat_inputs = 4
    neat_outputs = 13
    neat_hidden_layers = (4,)

    algorithm=HyperNEATFeedForwardLGNClass(
            substrate=LGNSubstrateLEO(
                layers=layers,
            ),
            neat=NEAT(
                pop_size=1000,
                species_size=20,
                survival_threshold=0.2,
                species_elitism=1,
                genome=DefaultGenomeCPPN(
                    num_inputs=neat_inputs,  # size of query coors
                    num_outputs=neat_outputs,
                    init_hidden_layers=neat_hidden_layers,
                    output_transform=ACT.sigmoid,
                ),
            ),
            activation=ACT.nand,
            output_transform=ACT.identity,
        )
    


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
        generation_limit=10000,
        seed=3
    )






    with open(f"results/{timestamp}_lgn/settings.txt", "w") as f_settings:
        f_settings.write(f"Generations: {pipeline.generation_limit}\n")
        f_settings.write(f"Fitness target: {pipeline.fitness_target}\n")
        f_settings.write(f"Population size: {algorithm.neat.pop_size}\n")
        f_settings.write(f"Substrate layers: {layers}\n")
        f_settings.write(f"Species: {algorithm.neat.species_controller.species_size}\n")
        f_settings.write(f"NEAT init. shape {neat_inputs, neat_hidden_layers, neat_outputs}\n")
        f_settings.write(f"Seed: {pipeline.seed}\n")
        f_settings.write(f"Problem task: {pipeline.problem.__class__.__name__}\n")
        f_settings.write(f"Elites: {algorithm.neat.species_controller.species_elitism}\n")
        f_settings.write(f"NEAT activation options {algorithm.neat.genome.node_gene.activation_options}\n")
        f_settings.write(f"NEAT aggregation options {algorithm.neat.genome.node_gene.aggregation_options}\n")
        f_settings.write(f"Survival threshold: {algorithm.neat.species_controller.survival_threshold}\n")





    print("Starting training ...")
    with open(f"results/{timestamp}_lgn/log.txt", "w") as f_log:
        with contextlib.redirect_stdout(f_log):

            # initialize state
            state = pipeline.setup()
            # print(state)
            # run until terminate

            filename = f"results/{timestamp}_lgn/current_gen.txt"

            state, best = pipeline.auto_run(state, filename)

            

    print(f"Finished training")

    with open(f"results/{timestamp}_lgn/best.txt", "w") as f_best:
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
            algorithm.neat.genome.visualize(network, save_path=f"results/{timestamp}_lgn/imgs/neat_CPPN_network.svg")

            transformed = algorithm.transform(state, best)
            seqs, h_nodes, h_conns, u_conns = transformed
            hyper_network = algorithm.hyper_genome.network_dict(state, h_nodes, h_conns)
            print(algorithm.hyper_genome.repr(state, h_nodes, h_conns))
            algorithm.hyper_genome.visualize(hyper_network, save_path=f"results/{timestamp}_lgn/imgs/hyperneat_network.svg")

    