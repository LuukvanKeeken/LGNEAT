import warnings

import jax
from jax import vmap, numpy as jnp
import numpy as np
import sympy as sp

from .default import DefaultGenome
from .gene import DefaultNode, DefaultConn
from .operations import DefaultMutation, DefaultCrossover, DefaultDistance
from .utils import unflatten_conns, extract_gene_attrs, extract_gene_attrs

from tensorneat.src.tensorneat.common import (
    topological_sort,
    topological_sort_python,
    find_useful_nodes,
    I_INF,
    attach_with_inf,
    ACT,
    AGG,
)


class DefaultGenomeArgmax(DefaultGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = "feedforward"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=50,
        max_conns=100,
        node_gene=DefaultNode(),
        conn_gene=DefaultConn(),
        mutation=DefaultMutation(),
        crossover=DefaultCrossover(),
        distance=DefaultDistance(),
        output_transform=None,
        input_transform=None,
        init_hidden_layers=(),
    ):

        super().__init__(
            num_inputs,
            num_outputs,
            max_nodes,
            max_conns,
            node_gene,
            conn_gene,
            mutation,
            crossover,
            distance,
            output_transform,
            input_transform,
            init_hidden_layers,
        )


    def forward(self, state, transformed, inputs):

        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        cal_seqs, nodes, conns, u_conns = transformed

        ini_vals = jnp.full((self.max_nodes,), jnp.nan)
        ini_vals = ini_vals.at[self.input_idx].set(inputs)
        nodes_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(self.node_gene, nodes)
        conns_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(self.conn_gene, conns)

        def cond_fun(carry):
            values, idx = carry
            return (idx < self.max_nodes) & (
                cal_seqs[idx] != I_INF
            )  # not out of bounds and next node exists

        def body_func(carry):
            values, idx = carry
            i = cal_seqs[idx]

            def input_node():
                return values

            def otherwise():
                # calculate connections
                conn_indices = u_conns[:, i]
                hit_attrs = attach_with_inf(
                    conns_attrs, conn_indices
                )  # fetch conn attrs
                ins = vmap(self.conn_gene.forward, in_axes=(None, 0, 0))(
                    state, hit_attrs, values
                )

                # calculate nodes
                z = self.node_gene.forward(
                    state,
                    nodes_attrs[i],
                    ins,
                    is_output_node=jnp.isin(
                        nodes[i, 0], self.output_idx
                    ),  # nodes[0] -> the key of nodes
                )

                # set new value
                new_values = values.at[i].set(z)
                return new_values

            values = jax.lax.cond(jnp.isin(i, self.input_idx), input_node, otherwise)

            return values, idx + 1

        vals, _ = jax.lax.while_loop(cond_fun, body_func, (ini_vals, 0))

        output_vals = vals[self.output_idx]

        if self.output_transform is None:
            output_vals = output_vals
        else:
            output_vals = self.output_transform(output_vals)

        return jnp.argmax(output_vals)


        if self.output_transform is None:
            return vals[self.output_idx]
        else:
            return self.output_transform(vals[self.output_idx])

    