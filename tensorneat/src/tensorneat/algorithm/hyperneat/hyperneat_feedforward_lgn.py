from typing import Callable, Sequence, Union, Optional

import jax
from jax import vmap, numpy as jnp

from tensorneat.src.tensorneat.algorithm.hyperneat.substrate import *
from tensorneat.src.tensorneat.algorithm.hyperneat.hyperneat import HyperNEAT, HyperNEATConn
from tensorneat.src.tensorneat.common import ACT, AGG, apply_activation_lgn, apply_aggregation_lgn, get_func_name
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
        output_transform: Callable = ACT.identity,
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
            node_gene=HyperNEATLGNNode(activation_default=activation),
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

        h_conns, h_nodes = self.keep_top2_per_postsynaptic(h_conns, h_nodes)

        h_nodes, h_conns = jax.device_put([h_nodes, h_conns])

        return self.hyper_genome.transform(state, h_nodes, h_conns)    


        # transformed = self.neat.transform(state, individual)
        # query_res = vmap(self.neat.forward, in_axes=(None, None, 0))(
        #     state, transformed, self.substrate.query_coors
        # )


        # h_nodes, h_conns = self.substrate.make_nodes(
        #     query_res
        # ), self.substrate.make_conns(query_res)

        # h_conns = self.keep_top2_per_postsynaptic(h_conns, h_nodes)

        # h_nodes, h_conns = jax.device_put([h_nodes, h_conns])

        # return self.hyper_genome.transform(state, h_nodes, h_conns)

    
    
    def keep_top2_per_postsynaptic(self, h_conns, h_nodes):
        post_ids = h_conns[:, 1]
        LEO_values = h_conns[:, 3]
        col4 = h_conns[:, 4]
        col5 = h_conns[:, 5]

        def process_post(post):
            mask = (post_ids == post)
            post_leos = jnp.where(mask, LEO_values, -jnp.inf)
            top2 = jnp.argsort(post_leos)[-2:]
            # sums = col4[top2] + col5[top2]
            nand_sum = jnp.sum(col4[top2])
            nor_sum = jnp.sum(col5[top2])
            sums = jnp.stack([nand_sum, nor_sum])
            argmax_idx = jnp.argmax(sums)
            top2_mask = jnp.zeros_like(LEO_values, dtype=bool).at[top2].set(True)
            return top2_mask & mask, argmax_idx

        posts = h_nodes[:, 0]
        all_masks, argmax_indices = jax.vmap(process_post)(posts)
        final_mask = jnp.any(all_masks, axis=0)
        h_conns = h_conns.at[:, 2].set(jnp.where(final_mask, 1.0, jnp.nan))
        # Set the second column of h_nodes to the argmax indices
        h_nodes = h_nodes.at[:, 1].set(argmax_indices)
        return h_conns, h_nodes
    

    # def keep_top2_per_postsynaptic_old(self, h_conns, h_nodes):
    #     post_ids = h_conns[:, 1]
    #     LEO_values = h_conns[:, 3]
    #     # Use h_nodes as the list of post-synaptic neuron indices
    #     def mask_for_post(post):
    #         mask = (post_ids == post)
    #         post_leos = jnp.where(mask, LEO_values, -jnp.inf)
    #         top2 = jnp.argsort(post_leos)[-2:]
    #         top2_mask = jnp.zeros_like(LEO_values, dtype=bool).at[top2].set(True)
    #         return top2_mask & mask

        

    #     all_masks = jax.vmap(mask_for_post)(h_nodes[:, 0])
    #     final_mask = jnp.any(all_masks, axis=0)
    #     h_conns = h_conns.at[:, 2].set(jnp.where(final_mask, 1.0, jnp.nan))
    #     return h_conns
    

class HyperNEATLGNNode(BaseNode):

    custom_attrs = ["activation_idx"]

    def __init__(
        self,
        activation_default: Optional[Callable] = None,
        activation_options: Union[Callable, Sequence[Callable]] = [ACT.nand, ACT.nor],
    ):
        super().__init__()
        
        
        if isinstance(activation_options, Callable):
            activation_options = [activation_options]

        if activation_default is None:
            activation_default = activation_options[0]

        self.activation_default = activation_options.index(activation_default)
        self.activation_options = activation_options
        self.activation_indices = jnp.arange(len(activation_options))

        # For LGN nodes, aggregation is always filter_nans
        self.aggregation_default = AGG.filter_nans


    def new_identity_attrs(self, state):

        activation_index = self.activation_default

        return jnp.array([activation_index])


    def new_random_attrs(self, state, randkey):

        # Randomly select activation function
        randkey, subkey = jax.random.split(randkey)
        activation_idx = jax.random.choice(subkey, self.activation_indices)

        return jnp.array([activation_idx])





    def forward(self, state, attrs, inputs, is_output_node=False):
        act = attrs[0]

        z = apply_aggregation_lgn(0, inputs, [self.aggregation_default])

        return apply_activation_lgn(act, z, self.activation_options)
    


    def repr(self, state, node, precision=2, idx_width=3, func_width=8):
        idx = node[0]
        act_func_idx = int(node[1])

        act_func = get_func_name(self.activation_options[act_func_idx])

        idx = int(idx)
        return "{}(idx={:<{idx_width}}, activation={:<{func_width}})".format(
            self.__class__.__name__, idx, act_func, idx_width=idx_width, func_width=func_width
        )
    
    def to_dict(self, state, node):
        node_idx, act_func_idx = node[:2]
        return {
            "idx": int(node_idx),
            "act_func_idx": int(act_func_idx),
        }
    

class HyperNEATLGNConn(HyperNEATConn):

    custom_attrs = ["weight"]

    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        in_idx, out_idx, weight, leo_value, nand, nor = conn

        in_idx = int(in_idx)
        out_idx = int(out_idx)
        weight = round(float(weight), precision)
        leo_value = round(float(leo_value), precision*2)
        nand = round(float(nand), precision*2)
        nor = round(float(nor), precision*2)

        return "{}(in: {:<{idx_width}}, out: {:<{idx_width}}, weight: {:<{float_width}}, leo: {:<{float_width}}, nand: {:<{float_width}}, nor: {:<{float_width}})".format(
            self.__class__.__name__,
            in_idx,
            out_idx,
            weight,
            leo_value,
            nand,
            nor,
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