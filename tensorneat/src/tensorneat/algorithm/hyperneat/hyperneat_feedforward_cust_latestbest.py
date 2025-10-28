"""
HyperNEAT with Feedforward Substrate and genome
"""

from typing import Callable

from tensorneat.src.tensorneat.algorithm.hyperneat.substrate import *
from tensorneat.src.tensorneat.algorithm.hyperneat.hyperneat import HyperNEAT, HyperNEATNode, HyperNEATConn
from tensorneat.src.tensorneat.algorithm.hyperneat.hyperneat_conn_improved import HyperNEATConnImproved
from tensorneat.src.tensorneat.common import ACT, AGG
from tensorneat.src.tensorneat.algorithm import NEAT
from tensorneat.src.tensorneat.genome import DefaultGenome


class HyperNEATFeedForwardCust(HyperNEAT):
    def __init__(
        self,
        substrate: BaseSubstrate,
        neat: NEAT,
        weight_threshold: float = 0.3,
        max_weight: float = 5.0,
        aggregation: Callable = AGG.sum,
        activation: Callable = ACT.sigmoid,
        output_transform: Callable = ACT.sigmoid,
    ):
        assert (
            substrate.query_coors.shape[1] == neat.num_inputs
        ), "Query coors of Substrate should be equal to NEAT input size"
        
        assert substrate.connection_type == "feedforward", "Substrate should be feedforward"

        self.substrate = substrate
        self.neat = neat
        self.weight_threshold = weight_threshold
        self.max_weight = max_weight
        self.hyper_genome = DefaultGenome(
            num_inputs=substrate.num_inputs,
            num_outputs=substrate.num_outputs,
            max_nodes=substrate.nodes_cnt,
            max_conns=substrate.conns_cnt,
            node_gene=HyperNEATNode(aggregation, activation),
            conn_gene=HyperNEATConnImproved(),
            output_transform=output_transform,
        )
        self.pop_size = neat.pop_size