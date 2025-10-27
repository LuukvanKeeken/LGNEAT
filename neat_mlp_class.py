import jax.numpy as jnp

from tensorneat.src.tensorneat.pipeline import Pipeline
from tensorneat.src.tensorneat.algorithm.neat import NEAT
from tensorneat.src.tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.src.tensorneat.problem.func_fit import CustomFuncFit
from tensorneat.src.tensorneat.common import ACT, AGG



# # define custom activate function and register it
# def square(x):
#     return x ** 2
# ACT.add_func("square", square)

if __name__ == "__main__":
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
        algorithm=NEAT(
            pop_size=5000,
            species_size=20,
            survival_threshold=0.01,
            genome=DefaultGenome(
                num_inputs=6,
                num_outputs=2,
                init_hidden_layers=(),
                node_gene=BiasNode(
                    activation_options=[ACT.identity, ACT.inv, ACT.square],
                    aggregation_options=[AGG.sum, AGG.product],
                ),
                output_transform=ACT.identity,
            ),
        ),
        problem=even_problem,
        generation_limit=20000,
        fitness_target=0.1,
        seed=42,
    )

    # initialize state
    state = pipeline.setup()
    # run until terminate
    state, best = pipeline.auto_run(state, "current_gen.txt")
    # show result
    pipeline.show(state, best)
