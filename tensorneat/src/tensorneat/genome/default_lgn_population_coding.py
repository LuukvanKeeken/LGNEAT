import warnings

import jax
from jax import vmap, numpy as jnp
import numpy as np
import sympy as sp

from .base import BaseGenome
from .gene import DefaultNode, DefaultConn
from .operations import DefaultMutation, DefaultCrossover, DefaultDistance
from .utils import unflatten_conns, extract_gene_attrs, extract_gene_attrs
from tensorneat.src.tensorneat.genome.default import DefaultGenome

from tensorneat.src.tensorneat.common import (
    topological_sort,
    topological_sort_python,
    find_useful_nodes,
    I_INF,
    attach_with_inf,
    ACT,
    AGG,
)

LABELS = {
    0: "nand",
    1: "nor",
    2: "and",
    3: "or",
    4: "false",
    5: "a & !b",
    6: "a",
    7: "xor",
    8: "xnor",
    9: "!a",
    10: "a | !b",
    11: "true",
    12: "!a & b",
    13: "b",
    14: "!b",
    15: "!a | b",
}

COLOURS = {
    0: "white",
    1: "gray",
    2: "green",
    3: "red",
    4: "black",
    5: "purple",
    6: "orange",
    7: "pink",
    8: "cyan",
    9: "brown",
    10: "magenta",
    11: "lightgray",
    12: "lightgreen",
    13: "lightblue",
    14: "lightcoral",
    15: "lightyellow",
}


class DefaultGenomeLGNPopulationCoding(DefaultGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = "feedforward"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_classes: int,
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

        self.num_classes = num_classes
        assert self.num_outputs % self.num_classes == 0, "num_outputs must be divisible by num_classes"


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

        # Group output_vals into self.num_classes groups and sum within each group
        output_vals = output_vals.reshape((self.num_classes, -1)).sum(axis=1)



        # Apply softmax to output_vals, regardless of output_transform
        exp_vals = jnp.exp(output_vals)
        output_vals = exp_vals / jnp.sum(exp_vals)

        return output_vals





    def visualize(
        self,
        network,
        rotate=0,
        reverse_node_order=False,
        size=(300, 300, 300),
        color=("yellow", "white", "blue", "gray"),
        with_labels=False,
        edgecolors="k",
        arrowstyle="->",
        arrowsize=3,
        edge_color=(0.3, 0.3, 0.3),
        save_path="network.svg",
        save_dpi=800,
        with_function_labels=False,
        make_compact=True,
        **kwargs,
    ):
        import networkx as nx
        from matplotlib import pyplot as plt

        conns_list = [k for k, v in network["conns"].items() if not jnp.isnan(v.get("weight"))]
        input_idx = self.get_input_idx()
        output_idx = self.get_output_idx()

        topo_order, topo_layers = network["topo_order"], network["topo_layers"]
        node2layer = {
            node: layer for layer, nodes in enumerate(topo_layers) for node in nodes
        }

        # reorder nodes in each layer to make them more compact
        subset_key = {}
        if make_compact:
            
            for layer, nodes in enumerate(topo_layers):
                if layer == 0 or len(nodes) == 1:
                    subset_key[layer] = nodes
                    continue
                nodes_y = []
                for node in nodes:
                    node_y = 0
                    for y, last_node in enumerate(topo_layers[layer-1]):
                        if (last_node, node) in conns_list:
                            node_y += y
                    nodes_y.append(node_y)
                nodes = [node for _, node in sorted(zip(nodes_y, nodes))]
                subset_key[layer] = nodes

        else:
            for layer, nodes in enumerate(topo_layers):
                subset_key[layer] = nodes

        if reverse_node_order:
            for layer, nodes in subset_key.items():
                subset_key[layer] = nodes[::-1]

        G = nx.DiGraph()

        if not isinstance(size, tuple):
            size = (size, size, size)
        if not isinstance(color, tuple):
            color = (color, color, color)

        labels = {}

        for node in topo_order:
            if node in input_idx:
                if with_labels:
                    labels[node] = network["nodes"][node].get("idx")

                G.add_node(node, subset=node2layer[node], size=size[0], color=color[0])
            elif node in output_idx:
                if with_labels:
                    if with_function_labels:
                        labels[node] = f"{network['nodes'][node].get('idx')}\n{LABELS.get(network['nodes'][node].get('act_func_idx'), 'unknown')}"
                    else:
                        labels[node] = f"{network['nodes'][node].get('idx')}"
                # G.add_node(node, subset=node2layer[node], size=size[2], color=color[2] if not (with_labels) else "gray")
                G.add_node(node, subset=node2layer[node], size=size[2], color=color[2] if not (with_labels) else (COLOURS.get(network['nodes'][node].get('act_func_idx'), color[2]) if not with_function_labels else "gray"))
            else:
                # Colour the hidden nodes based on their activation function
                if network["nodes"][node].get("idx") == node:
                    if with_labels:
                        if with_function_labels:
                            labels[node] = f"{network['nodes'][node].get('idx')}\n{LABELS.get(network['nodes'][node].get('act_func_idx'), 'unknown')}"
                        else:
                            labels[node] = f"{network['nodes'][node].get('idx')}"


                    G.add_node(node, subset=node2layer[node], size=size[1], color=COLOURS.get(network['nodes'][node].get('act_func_idx'), 'white') if not (with_function_labels) else "white")
                    # if network["nodes"][node].get("act_func_idx") == 0: # nand
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="white")
                    # elif network["nodes"][node].get("act_func_idx") == 1: # nor
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="gray" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 2: # and
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="green" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 3: # or
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="red" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 4: # false
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="black" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 5: # a_and_not_b
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="purple" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 6: # a
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="orange" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 7: # xor
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="pink" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 8: # xnor
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="cyan" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 9: # not_a
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="brown" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 10: # a_or_not_b
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="magenta" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 11: # true
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="lightgray" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 12: # not_a_and_b
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="lightgreen" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 13: # b
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="lightblue" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 14: # not_b
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="lightcoral" if not with_function_labels else "white")
                    # elif network["nodes"][node].get("act_func_idx") == 15: # not_a_or_b
                    #     G.add_node(node, subset=node2layer[node], size=size[1], color="lightyellow" if not with_function_labels else "white")
                else:
                    raise ValueError("Node idx does not match the key in network['nodes']")



        for conn in conns_list:
            G.add_edge(conn[0], conn[1])
        pos = nx.multipartite_layout(G, subset_key=subset_key)


        def rotate_layout(pos, angle):
            angle_rad = np.deg2rad(angle)
            cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
            rotated_pos = {}
            for node, (x, y) in pos.items():
                rotated_pos[node] = (
                    cos_angle * x - sin_angle * y,
                    sin_angle * x + cos_angle * y,
                )
            return rotated_pos

        rotated_pos = rotate_layout(pos, rotate)

        node_sizes = [n["size"] for n in G.nodes.values()]
        node_colors = [n["color"] for n in G.nodes.values()]

        font_size = None
        if with_labels:
            font_size = 8
            if with_function_labels:
                font_size = 6

        nx.draw(
            G,
            pos=rotated_pos,
            node_size=node_sizes,
            node_color=node_colors,
            with_labels=with_labels,
            edgecolors=edgecolors,
            arrowstyle=arrowstyle,
            arrowsize=arrowsize,
            edge_color=edge_color,
            labels=labels if with_labels else None,
            font_size=font_size,
        )
        plt.savefig(save_path, dpi=save_dpi)
        plt.close()
