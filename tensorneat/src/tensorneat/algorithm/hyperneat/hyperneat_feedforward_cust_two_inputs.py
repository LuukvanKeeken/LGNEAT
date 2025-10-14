from typing import Callable

import jax
from jax import vmap, numpy as jnp

from tensorneat.src.tensorneat.algorithm.hyperneat.substrate import *
from tensorneat.src.tensorneat.algorithm.hyperneat.hyperneat import HyperNEAT, HyperNEATNode, HyperNEATConn
from tensorneat.src.tensorneat.algorithm.hyperneat.hyperneat_conn_improved import HyperNEATConnImproved
from tensorneat.src.tensorneat.common import ACT, AGG
from tensorneat.src.tensorneat.algorithm import NEAT
from tensorneat.src.tensorneat.genome.default_two_inputs import DefaultGenomeTwoInputs


class HyperNEATFeedForwardCustTwoInputs(HyperNEAT):
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
        self.hyper_genome = DefaultGenomeTwoInputs(
            num_inputs=substrate.num_inputs,
            num_outputs=substrate.num_outputs,
            max_nodes=substrate.nodes_cnt,
            max_conns=substrate.conns_cnt,
            node_gene=HyperNEATNode(aggregation, activation),
            conn_gene=HyperNEATConnImproved(),
            output_transform=output_transform,
        )
        self.pop_size = neat.pop_size


    def transform(self, state, individual):
        transformed = self.neat.transform(state, individual)
        query_res = vmap(self.neat.forward, in_axes=(None, None, 0))(
            state, transformed, self.substrate.query_coors
        )
        
        # make query res in range [-max_weight, max_weight]
        query_res = jnp.where(
            query_res > 0, query_res - self.weight_threshold, query_res
        )
        query_res = jnp.where(
            query_res < 0, query_res + self.weight_threshold, query_res
        )
        query_res = query_res / (1 - self.weight_threshold) * self.max_weight

        h_nodes, h_conns = self.substrate.make_nodes(
            query_res
        ), self.substrate.make_conns(query_res)

        h_conns = self.keep_top2_per_postsynaptic(h_conns, h_nodes)

        h_nodes, h_conns = jax.device_put([h_nodes, h_conns])

        return self.hyper_genome.transform(state, h_nodes, h_conns)

    
    
    def keep_top2_per_postsynaptic(self, h_conns, h_nodes):
        post_ids = h_conns[:, 1]
        weights = h_conns[:, 2]
        # Use h_nodes as the list of post-synaptic neuron indices
        def mask_for_post(post):
            mask = (post_ids == post)
            post_weights = jnp.where(mask, weights, -jnp.inf)
            top2 = jnp.argsort(post_weights)[-2:]
            top2_mask = jnp.zeros_like(weights, dtype=bool).at[top2].set(True)
            return top2_mask & mask

        all_masks = jax.vmap(mask_for_post)(h_nodes)
        final_mask = jnp.any(all_masks, axis=0)
        h_conns = h_conns.at[:, 2].set(jnp.where(final_mask, h_conns[:, 2], 0.0))
        return h_conns