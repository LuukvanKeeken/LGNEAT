from typing import Callable

import jax
from jax import vmap, numpy as jnp

from tensorneat.src.tensorneat.algorithm.hyperneat.substrate import *
from tensorneat.src.tensorneat.algorithm.hyperneat.hyperneat import HyperNEAT, HyperNEATConn
from tensorneat.src.tensorneat.common import ACT, AGG
from tensorneat.src.tensorneat.algorithm import NEAT
from tensorneat.src.tensorneat.genome import DefaultGenomeLGN, BaseNode


class HyperNEATFeedForwardLGN(HyperNEAT):
    def __init__(
        self,
        substrate: BaseSubstrate,
        neat: NEAT,
        weight_threshold: float = 0.3,
        max_weight: float = 5.0,
        aggregation: Callable = AGG.filter_nans,
        activation: Callable = ACT.nand,
        output_transform: Callable = ACT.nand,
    ):
        assert (
            substrate.query_coors.shape[1] == neat.num_inputs
        ), "Query coors of Substrate should be equal to NEAT input size"
        
        assert substrate.connection_type == "feedforward", "Substrate should be feedforward"

        self.substrate = substrate
        self.neat = neat
        self.weight_threshold = weight_threshold
        self.max_weight = max_weight
        self.hyper_genome = DefaultGenomeLGN(
            num_inputs=substrate.num_inputs,
            num_outputs=substrate.num_outputs,
            max_nodes=substrate.nodes_cnt,
            max_conns=substrate.conns_cnt,
            node_gene=HyperNEATLGNNode(aggregation, activation),
            conn_gene=HyperNEATLGNConn(),
            output_transform=output_transform,
        )
        self.pop_size = neat.pop_size


    @property
    def num_inputs(self):
        return self.substrate.num_inputs
    

    def forward(self, state, transformed, inputs):
        # Don't use bias
        res = self.hyper_genome.forward(state, transformed, inputs)
        return res


    def transform(self, state, individual):
        transformed = self.neat.transform(state, individual)
        query_res = vmap(self.neat.forward, in_axes=(None, None, 0))(
            state, transformed, self.substrate.query_coors
        )


        h_nodes, h_conns = self.substrate.make_nodes(
            query_res
        ), self.substrate.make_conns(query_res)

        h_conns = self.keep_top2_per_postsynaptic(h_conns, h_nodes)

        h_nodes, h_conns = jax.device_put([h_nodes, h_conns])

        return self.hyper_genome.transform(state, h_nodes, h_conns)

    
    
    def keep_top2_per_postsynaptic(self, h_conns, h_nodes):
        post_ids = h_conns[:, 1]
        LEO_values = h_conns[:, 3]
        # Use h_nodes as the list of post-synaptic neuron indices
        def mask_for_post(post):
            mask = (post_ids == post)
            post_leos = jnp.where(mask, LEO_values, -jnp.inf)
            top2 = jnp.argsort(post_leos)[-2:]
            top2_mask = jnp.zeros_like(LEO_values, dtype=bool).at[top2].set(True)
            return top2_mask & mask

        

        all_masks = jax.vmap(mask_for_post)(h_nodes)
        final_mask = jnp.any(all_masks, axis=0)
        h_conns = h_conns.at[:, 2].set(jnp.where(final_mask, 1.0, jnp.nan))
        return h_conns
    

class HyperNEATLGNNode(BaseNode):
    def __init__(
        self,
        aggregation=AGG.filter_nans,
        activation=ACT.nand,
    ):
        super().__init__()
        self.aggregation = aggregation
        self.activation = activation

    def forward(self, state, attrs, inputs, is_output_node=False):
        return self.activation(self.aggregation(inputs))
    

class HyperNEATLGNConn(HyperNEATConn):

    custom_attrs = ["weight"]

    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        in_idx, out_idx, weight, leo_value = conn

        in_idx = int(in_idx)
        out_idx = int(out_idx)
        weight = round(float(weight), precision)
        leo_value = round(float(leo_value), precision)

        return "{}(in: {:<{idx_width}}, out: {:<{idx_width}}, weight: {:<{float_width}}, leo: {:<{float_width}})".format(
            self.__class__.__name__,
            in_idx,
            out_idx,
            weight,
            leo_value,
            idx_width=idx_width,
            float_width=precision + 3,
        )

    def to_dict(self, state, conn):
        in_idx, out_idx, weight, leo_value = conn[:4]
        return {
            "in": int(in_idx),
            "out": int(out_idx),
            "weight": float(weight),
            "leo": float(leo_value),
        }