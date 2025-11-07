import warnings

import jax
from jax import vmap, numpy as jnp
import numpy as np
import sympy as sp

from .default import DefaultGenome
from .gene import DefaultNode, DefaultConn, OriginNode, OriginConn
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

# Map function names to their actual functions and abbreviations
ACTS = {
    "gaussian": [ACT.gaussian, "gau"],
    "sigmoid": [ACT.sigmoid, "sig"],
    "sin": [ACT.sin, "sin"],
    "abs": [ACT.abs, "abs"],
}


LABELS = {
    0: "LEO",
    1: "nand",
    2: "nor",
    3: "and",
    4: "or",
    5: "false",
    6: "a & !b",
    7: "a",
    8: "xor",
    9: "xnor",
    10: "!a",
    11: "a | !b",
    12: "true",
    13: "!a & b",
    14: "b",
    15: "!b",
    16: "!a | b",
}


class DefaultGenomeCPPN(DefaultGenome):

    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = "feedforward"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=50,
        max_conns=100,
        # node_gene=DefaultNode(activation_options=[v[0] for v in ACTS.values()]),
        # conn_gene=DefaultConn(),
        node_gene=OriginNode(activation_options=[v[0] for v in ACTS.values()]),
        conn_gene=OriginConn(),
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


    def visualize(
        self,
        network,
        rotate=0,
        reverse_node_order=False,
        size=(300, 300, 300),
        color=("yellow", "white", "blue"),
        with_labels=False,
        with_function_labels=False,
        edgecolors="k",
        arrowstyle="->",
        arrowsize=3,
        edge_color=(0.3, 0.3, 0.3),
        save_path="network.svg",
        save_dpi=800,
        **kwargs,
    ):
        import networkx as nx
        from matplotlib import pyplot as plt

        conns_list = list(network["conns"])
        input_idx = self.get_input_idx()
        output_idx = self.get_output_idx()

        topo_order, topo_layers = network["topo_order"], network["topo_layers"]
        node2layer = {
            node: layer for layer, nodes in enumerate(topo_layers) for node in nodes
        }

        # reorder nodes in each layer to make them more compact
        subset_key = {}
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
                    labels[node] = f"{network['nodes'][node].get('idx')}"

                G.add_node(node, subset=node2layer[node], size=size[0], color=color[0])
            elif node in output_idx:
                if with_labels:
                    node_idx = network['nodes'][node].get('idx')
                    node_idx -= min(output_idx)
                    if with_function_labels:
                        labels[node] = f"{LABELS.get(node_idx, 'unk')}\n{ACTS.get(network['nodes'][node].get('act'), 'unk')[1]}"
                    else:
                        labels[node] = f"{LABELS.get(node_idx, 'unk')}"

                G.add_node(node, subset=node2layer[node], size=size[2], color=color[2] if not with_labels else "gray")
            else:
                if with_labels:
                    if with_function_labels:
                        labels[node] = f"{network['nodes'][node].get('idx')}\n{ACTS.get(network['nodes'][node].get('act'), 'unk')[1]}"
                    else:
                        labels[node] = f"{network['nodes'][node].get('idx')}"

                G.add_node(node, subset=node2layer[node], size=size[1], color=color[1])

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