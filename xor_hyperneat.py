from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate
from tensorneat.genome import DefaultGenome
from tensorneat.common import ACT

from tensorneat.problem.func_fit import XOR3d

if __name__ == "__main__":

    algorithm=HyperNEAT(
            substrate=FullSubstrate(
                input_coors=((-1, -1), (-0.33, -1), (0.33, -1), (1, -1)),
                hidden_coors=((-1, 0), (0, 0), (1, 0)),
                output_coors=((0, 1),),
            ),
            neat=NEAT(
                pop_size=10000,
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
            activate_time=10,
            output_transform=ACT.sigmoid,
        )
    
    pipeline = Pipeline(
        algorithm=algorithm,
        problem=XOR3d(),
        generation_limit=200
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)

    # visualize the best individual
    network = algorithm.neat.genome.network_dict(state, *best)
    print(algorithm.neat.genome.repr(state, *best))
    algorithm.neat.genome.visualize(network, save_path="./imgs/xor_hyperneat_network.svg")
