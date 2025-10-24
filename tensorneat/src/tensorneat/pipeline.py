import json
import os
import warnings

import jax, jax.numpy as jnp
import datetime, time
import numpy as np

from tensorneat.src.tensorneat.algorithm import BaseAlgorithm
from tensorneat.src.tensorneat.problem import BaseProblem
from tensorneat.src.tensorneat.common import State, StatefulBaseClass


class Pipeline(StatefulBaseClass):
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        problem: BaseProblem,
        seed: int = 42,
        fitness_target: float = 1,
        generation_limit: int = 1000,
        is_save: bool = False,
        save_dir=None,
        show_problem_details: bool = False,
        using_multidevice: bool = False,
    ):
        assert problem.jitable, "Currently, problem must be jitable"

        self.algorithm = algorithm
        self.problem = problem
        self.seed = seed
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.pop_size = self.algorithm.pop_size

        np.random.seed(self.seed)

        assert (
            algorithm.num_inputs == self.problem.input_shape[-1]
        ), f"algorithm input shape is {algorithm.num_inputs} but problem input shape is {self.problem.input_shape}"

        self.best_genome = None
        self.best_fitness = float("-inf")
        self.generation_timestamp = None
        self.is_save = is_save

        if is_save:
            if save_dir is None:
                now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                self.save_dir = f"./{self.__class__.__name__} {now}"
            else:
                self.save_dir = save_dir
            print(f"save to {self.save_dir}")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.genome_dir = os.path.join(self.save_dir, "genomes")
            if not os.path.exists(self.genome_dir):
                os.makedirs(self.genome_dir)

        self.show_problem_details = show_problem_details

        self.using_multidevice = using_multidevice
        if self.using_multidevice:
            assert jax.device_count() > 1, f"using_multidevice requires more than 1 device, but {jax.device_count()=} devices are available"
            print(f"Using {jax.device_count()} devices!")

    def setup(self, state=State()):
        print("initializing")
        state = state.register(randkey=jax.random.PRNGKey(self.seed))

        state = self.algorithm.setup(state)
        state = self.problem.setup(state)

        if self.is_save:
            # self.save(state=state, path=os.path.join(self.save_dir, "pipeline.pkl"))
            with open(os.path.join(self.save_dir, "config.txt"), "w") as f:
                f.write(json.dumps(self.show_config(), indent=4))
            # create log file
            with open(os.path.join(self.save_dir, "log.txt"), "w") as f:
                f.write("Generation,Max,Min,Mean,Std,Cost Time\n")

        print("initializing finished")
        return state

    def step(self, state):
        """
        returns: 
            state, previous_pop, fitnesses
        state: updated state
        previous_pop: previous population
        fitnesses: fitnesses of previous population
        """

        randkey_, randkey = jax.random.split(state.randkey)

        pop = self.algorithm.ask(state)

        pop_transformed = jax.vmap(self.algorithm.transform, in_axes=(None, 0))(
            state, pop
        )

        if not self.using_multidevice:
            keys = jax.random.split(randkey_, self.pop_size)
            fitnesses = jax.vmap(self.problem.evaluate, in_axes=(None, 0, None, 0))(
                state, keys, self.algorithm.forward, pop_transformed
            )
        else: # using_multidevice
            num_devices = jax.device_count()
            assert self.pop_size % num_devices == 0, "if you want to use multiple gpus, pop_size must be divisible by jax.device_count()"
            pop_size_per_device = self.pop_size // num_devices

            keys = jax.random.split(randkey_, (num_devices, pop_size_per_device))
            split_pop_transformed = jax.tree_map(
                lambda x: x.reshape(num_devices, pop_size_per_device, *x.shape[1:]),
                pop_transformed
            )

            fitnesses = jax.pmap(
                lambda key_slice, pop_slice: jax.vmap(self.problem.evaluate, in_axes=(None, 0, None, 0))(
                    state, key_slice, self.algorithm.forward, pop_slice
                ),
                axis_name='devices',
                in_axes=(0, 0)
            )(keys, split_pop_transformed)

            fitnesses = fitnesses.reshape(self.pop_size)

        # replace nan with -inf
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        previous_pop = self.algorithm.ask(state)
        state = self.algorithm.tell(state, fitnesses)

        return state.update(randkey=randkey), previous_pop, fitnesses

    def auto_run(self, state, filename):
        print("start compile")
        tic = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                message=r"The jitted function .* includes a pmap. Using jit-of-pmap can lead to inefficient data movement"
            )
            compiled_step = jax.jit(self.step).lower(state).compile()

        if self.show_problem_details:
            self.compiled_pop_transform_func = (
                jax.jit(jax.vmap(self.algorithm.transform, in_axes=(None, 0)))
                .lower(state, self.algorithm.ask(state))
                .compile()
            )

        print(
            f"compile finished, cost time: {time.time() - tic:.6f}s",
        )

        with open(filename, "w") as f_log:

            for gen in range(self.generation_limit):

                self.generation_timestamp = time.time()

                state, previous_pop, fitnesses = compiled_step(state)

                fitnesses = jax.device_get(fitnesses)

                self.analysis(state, previous_pop, fitnesses)

                if max(fitnesses) >= self.fitness_target:
                    print("Fitness limit reached!")
                    print(f"{jnp.max(fitnesses)} at {jnp.argmax(fitnesses)}")
                    # print best genome
                    best_fitness = self.best_fitness
                    best_genome = self.best_genome
                    print(f"Best genome fitness: {best_fitness}")
                    print(f"Best genome: {best_genome}")

                    for fit in range(len(fitnesses)):
                        print(f"Genome {fit} fitness: {fitnesses[fit]}")
                    break

                # overwrite the previous progress line instead of appending
                # move to file start, write the line, then truncate the rest
                if (gen+1)%100 == 0:
                    f_log.seek(0)
                    f_log.write(f"Generation {gen+1}/{self.generation_limit}\n")
                    f_log.truncate()
                    f_log.flush()

        if int(state.generation) >= self.generation_limit:
            print("Generation limit reached!")

        if self.is_save:
            best_genome = jax.device_get(self.best_genome)
            with open(os.path.join(self.genome_dir, f"best_genome.npz"), "wb") as f:
                np.savez(
                    f,
                    nodes=best_genome[0],
                    conns=best_genome[1],
                    fitness=self.best_fitness,
                )

        # Do fitness evaluation of previous_pop
        previous_pop_transformed = jax.vmap(self.algorithm.transform, in_axes=(None, 0))(
            state, previous_pop
        )
        fitnesses_previous_pop = jax.vmap(self.problem.evaluate, in_axes=(None, 0, None, 0))(
                state, None, self.algorithm.forward, previous_pop_transformed
            )
        
        print(f"Again fitnesses:")
        print(f"{jnp.max(fitnesses_previous_pop)} at {jnp.argmax(fitnesses_previous_pop)} ")
        print(f"fitness at index 414: {fitnesses_previous_pop[414]}")
        print("all fitness previous pop:")
        for fit in range(len(fitnesses_previous_pop)):
            print(f"Genome {fit} fitness: {fitnesses_previous_pop[fit]}")

        return state, self.best_genome

    def analysis(self, state, pop, fitnesses):

        generation = int(state.generation)

        valid_fitnesses = fitnesses[~np.isinf(fitnesses)]
        # avoid there is no valid fitness in the whole population
        if len(valid_fitnesses) == 0:
            max_f, min_f, mean_f, std_f = ["NaN"] * 4
        else:
            max_f, min_f, mean_f, std_f = (
                max(valid_fitnesses),
                min(valid_fitnesses),
                np.mean(valid_fitnesses),
                np.std(valid_fitnesses),
            )

        new_timestamp = time.time()

        cost_time = new_timestamp - self.generation_timestamp

        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = fitnesses[max_idx]
            self.best_genome = pop[0][max_idx], pop[1][max_idx]
            print("Recalculate fitness of new best genome at this point")
            recalculated_fitness = self.problem.evaluate(
                state, None, self.algorithm.forward,
                self.algorithm.transform(state, self.best_genome)
            )
            print(f"Recalculated fitness: {recalculated_fitness}")
            print(f"self.best_fitness: {self.best_fitness}")
            print(f"max_idx: {max_idx}")
            if not jnp.isclose(recalculated_fitness, self.best_fitness, atol=1e-4):
                raise ValueError(f"Recalculated fitness {recalculated_fitness} does not match stored best fitness {self.best_fitness}!")


        if self.is_save:
            # save best
            best_genome = jax.device_get((pop[0][max_idx], pop[1][max_idx]))
            file_name = os.path.join(self.genome_dir, f"{generation}.npz")
            with open(file_name, "wb") as f:
                np.savez(
                    f,
                    nodes=best_genome[0],
                    conns=best_genome[1],
                    fitness=self.best_fitness,
                )

            # append log
            with open(os.path.join(self.save_dir, "log.txt"), "a") as f:
                f.write(f"{generation},{max_f},{min_f},{mean_f},{std_f},{cost_time}\n")

        print(
            f"Generation: {generation}, Cost time: {cost_time * 1000:.2f}ms\n",
            f"\tfitness: valid cnt: {len(valid_fitnesses)}, max: {max_f:.4f}, best: {self.best_fitness:.4f}, min: {min_f:.4f}, mean: {mean_f:.4f}, std: {std_f:.4f}\n",
        )

        self.algorithm.show_details(state, fitnesses)

        if self.show_problem_details:
            pop_transformed = self.compiled_pop_transform_func(
                state, pop #using previous pop instead of requesting the new one from state here
            )
            self.problem.show_details(
                state, state.randkey, self.algorithm.forward, pop_transformed
            )
        # show details for problem

    def show(self, state, best, *args, **kwargs):
        transformed = self.algorithm.transform(state, best)
        return self.problem.show(
            state, state.randkey, self.algorithm.forward, transformed, *args, **kwargs
        )
