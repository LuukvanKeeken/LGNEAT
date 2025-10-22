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
    # Check if input coordinates are inside a circle
    # Return [1, 0] if inside, else [0, 1]
    def inside_circle(inputs, radius=0.5):
        x, y = inputs
        res = jnp.square(x) + jnp.square(y)
        
        return jnp.where(res <= radius**2, jnp.array([1, 0]), jnp.array([0, 1]))
    
    inside_circle_problem = CustomFuncFit(
        func = inside_circle,
        low_bounds = [-1, -1],
        upper_bounds = [1, 1],
        method = "sample",
        num_samples = 20
    )

    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=10000,
            species_size=20,
            survival_threshold=0.01,
            genome=DefaultGenome(
                num_inputs=2,
                num_outputs=1,
                init_hidden_layers=(),
                node_gene=BiasNode(
                    activation_options=[ACT.identity, ACT.inv, ACT.square],
                    aggregation_options=[AGG.sum, AGG.product],
                ),
                output_transform=ACT.identity,
            ),
        ),
        problem=inside_circle_problem,
        generation_limit=50,
        fitness_target=-1e-4,
        seed=42,
    )

    # initialize state
    state = pipeline.setup()
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)
