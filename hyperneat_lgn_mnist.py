from tensorneat.src.tensorneat.pipeline import Pipeline
from tensorneat.src.tensorneat.algorithm.neat import NEAT
from tensorneat.src.tensorneat.algorithm.hyperneat import LGNSubstrateLEO
from tensorneat.src.tensorneat.algorithm.hyperneat import HyperNEATFeedForwardLGNClassSeq
from tensorneat.src.tensorneat.genome import DefaultGenomeCPPN
from tensorneat.src.tensorneat.genome.operations import DefaultMutation
from tensorneat.src.tensorneat.common import ACT
import jax.numpy as jnp
from tensorneat.src.tensorneat.problem.func_fit import CustomFuncFit
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets

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


if not os.path.exists("datasets"):
    os.makedirs("datasets")


if not os.path.exists(f"datasets/MNIST"):
    torchvision.datasets.MNIST("datasets", download=True)

include_classes = [9, 5]
num_train_per_class = 50
num_test_per_class = 50
train_set = torchvision.datasets.MNIST("datasets", train=True)
test_set = torchvision.datasets.MNIST("datasets", train=False)


def pil_to_jax_array_and_binarize(pil_img):
    np_img = np.array(pil_img, dtype=np.float32)
    # Now binarize the image: pixels > 127 become 1, else 0
    np_img = (np_img > 127).astype(np.float32)
    return jnp.array(np_img)

def label_to_one_hot(label):
    one_hot = jnp.zeros(len(include_classes), dtype=jnp.float32)
    index = include_classes.index(label)
    one_hot = one_hot.at[index].set(1.0)
    return one_hot


# Initialize counters and storage
train_class_counts = {cls: 0 for cls in include_classes}
train_images, train_labels = [], []

for img, label in train_set:
    if label in include_classes and train_class_counts[label] < num_train_per_class:
        train_images.append(pil_to_jax_array_and_binarize(img))
        train_labels.append(label_to_one_hot(label))
        train_class_counts[label] += 1
    # Stop early if all classes are filled
    if all(count == num_train_per_class for count in train_class_counts.values()):
        break

# Repeat for test set
test_class_counts = {cls: 0 for cls in include_classes}
test_images, test_labels = [], []

for img, label in test_set:
    if label in include_classes and test_class_counts[label] < num_test_per_class:
        test_images.append(pil_to_jax_array_and_binarize(img))
        test_labels.append(label_to_one_hot(label))
        test_class_counts[label] += 1
    if all(count == num_test_per_class for count in test_class_counts.values()):
        break

# Make some quick plots of the first 5 train images
# plt.figure(figsize=(10, 2))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(train_images[i], cmap="gray")
#     plt.title(f"Label: {train_labels[i]}")
#     plt.axis("off")
# plt.savefig(f"results/{timestamp}_lgn/imgs/sample_train_images.png")

if __name__ == "__main__":

    all_gen_nums = []
    all_max_fit = []
    all_mean_fit = []
    num_runs = 1
    for i in range(num_runs):

        layers = [28, 30, 30, 20, 10]

        neat_inputs = 4
        neat_outputs = 17
        neat_hidden_layers = (4,)

        algorithm=HyperNEATFeedForwardLGNClassSeq(
                substrate=LGNSubstrateLEO(
                    layers=layers,
                ),
                neat=NEAT(
                    pop_size=100,
                    species_size=8,
                    survival_threshold=0.1,
                    species_elitism=1,
                    compatibility_threshold=1.0,
                    genome=DefaultGenomeCPPN(
                        num_inputs=neat_inputs,  # size of query coors
                        num_outputs=neat_outputs,
                        init_hidden_layers=neat_hidden_layers,
                        output_transform=ACT.identity,
                        mutation=DefaultMutation(),
                    ),
                ),
                activation=ACT.nand,
                output_transform=ACT.identity,
            )
        


      

        mnist = CustomFuncFit(
            func = None,
            low_bounds = jnp.zeros((28, 28)),
            upper_bounds = jnp.ones((28, 28)) * 1.1,
            method = "set_direct",
            step_size = jnp.ones((28, 28)),
            train_test_split=0.8,
            split_seed=i,
            input_data = jnp.array(train_images),
            output_data = jnp.array(train_labels),
        )

        pipeline = Pipeline(
            algorithm=algorithm,
            problem=mnist,
            fitness_target=-0.02,
            generation_limit=100,
            seed=i,
            is_save=True,
            save_dir=f"results/{timestamp}_lgn",
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
            f_settings.write(f"Train/test split: {pipeline.problem.train_test_split}\n")
            f_settings.write(f"Split seed: {pipeline.problem.split_seed}\n")
            f_settings.write(f"Compatibility threshold: {algorithm.neat.species_controller.compatibility_threshold}\n")





        print("Starting training ...")
        start_train_time = time.time()
        with open(f"results/{timestamp}_lgn/log_prints.txt", "w") as f_log:
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

                # To get the approximate number of actually completed generations,
                # read out the first number in the first line of current_gen.txt
                # First check if the file actually exists.
                completed_generations = pipeline.generation_limit
                if os.path.exists(f"results/{timestamp}_lgn/current_gen.txt"):
                    with open(f"results/{timestamp}_lgn/current_gen.txt", "r") as f:
                        first_line = f.readline()
                        # Check if the line is not empty
                        if first_line:
                            completed_generations = int(first_line.split("/")[0].split(" ")[1])
                all_gen_nums.append(completed_generations)

                print(f"Approximate time/generation: {(time.time() - start_train_time)/completed_generations:.2f} seconds\n")

                start_time_test = time.time()
                # show result
                pipeline.show(state, best, leave_out_input=True)
                print(f"Testing time: {time.time() - start_time_test} seconds\n")

                # visualize the best individual
                network = algorithm.neat.genome.network_dict(state, *best)
                print(algorithm.neat.genome.repr(state, *best))
                algorithm.neat.genome.visualize(network, save_path=f"results/{timestamp}_lgn/imgs/neat_CPPN_network_nofuncs.svg", with_labels=True)
                algorithm.neat.genome.visualize(network, save_path=f"results/{timestamp}_lgn/imgs/neat_CPPN_network.svg", with_labels=True, with_function_labels=True)

                transformed = algorithm.transform(state, best)
                seqs, h_nodes, h_conns, u_conns = transformed
                hyper_network = algorithm.hyper_genome.network_dict(state, h_nodes, h_conns)
                print(algorithm.hyper_genome.repr(state, h_nodes, h_conns))
                algorithm.hyper_genome.visualize(hyper_network, save_path=f"results/{timestamp}_lgn/imgs/hyperneat_network_nofuncs_comp.svg", with_labels=True)
                algorithm.hyper_genome.visualize(hyper_network, save_path=f"results/{timestamp}_lgn/imgs/hyperneat_network_comp.svg", with_labels=True, with_function_labels=True)
                algorithm.hyper_genome.visualize(hyper_network, save_path=f"results/{timestamp}_lgn/imgs/hyperneat_network_nofuncs.svg", with_labels=True, make_compact=False)
                algorithm.hyper_genome.visualize(hyper_network, save_path=f"results/{timestamp}_lgn/imgs/hyperneat_network.svg", with_labels=True, with_function_labels=True, make_compact=False)
        
        del pipeline, algorithm


        # Plot the progression of the max and meand/std fitness over generations
        # The data is in results/timestamp/log.txt, with the relevant data in 
        # the second (max), fourth (mean) and fifth (std) columns. The first
        # line is a header.
        log_data = np.genfromtxt(f"results/{timestamp}_lgn/log.txt", skip_header=1, delimiter=',')
        generations = log_data[:, 0]
        max_fitness = log_data[:, 1]
        mean_fitness = log_data[:, 3]
        std_fitness = log_data[:, 4]

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(generations, max_fitness, label="Max Fitness")
        plt.plot(generations, mean_fitness, label="Mean Fitness")
        plt.fill_between(generations, mean_fitness - std_fitness, mean_fitness + std_fitness, alpha=0.2, label="Std Fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Fitness Progression")
        plt.legend()
        plt.savefig(f"results/{timestamp}_lgn/imgs/fitness_progression.png")
        plt.close()

        avg_gens = sum(all_gen_nums)/len(all_gen_nums)
        stddev_gens = np.std(np.array(all_gen_nums))
        all_max_fit.append(max_fitness[-1])
        all_mean_fit.append(mean_fitness[-1])
        avg_max_fit = sum(all_max_fit)/len(all_max_fit)
        avg_mean_fit = sum(all_mean_fit)/len(all_mean_fit)
        stddev_max_fit = np.std(np.array(all_max_fit))
        stddev_mean_fit = np.std(np.array(all_mean_fit))
        print(f"Average generations over {i+1} runs: {avg_gens}+/-{stddev_gens}")
        print(f"All generation counts: {all_gen_nums}")
        print(f"Average max fitness over {i+1} runs: {avg_max_fit}+/-{stddev_max_fit}")
        print(f"All max fitnesses: {all_max_fit}")
        print(f"Average mean fitness over {i+1} runs: {avg_mean_fit}+/-{stddev_mean_fit}")
        print(f"All mean fitnesses: {all_mean_fit}")
        with open(f"results/{timestamp}_lgn/avg_gens.txt", "w") as f_avg:
            f_avg.write(f"Average generations over {i+1} runs: {avg_gens}+/-{stddev_gens}\n")
            f_avg.write(f"All generation counts: {all_gen_nums}\n")
            f_avg.write(f"Average max fitness over {i+1} runs: {avg_max_fit}+/-{stddev_max_fit}\n")
            f_avg.write(f"All max fitnesses: {all_max_fit}\n")
            f_avg.write(f"Average mean fitness over {i+1} runs: {avg_mean_fit}+/-{stddev_mean_fit}\n")
            f_avg.write(f"All mean fitnesses: {all_mean_fit}\n")

    