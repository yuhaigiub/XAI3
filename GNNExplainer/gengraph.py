"""
gengraph.py
Generating and manipulaton the synthetic graphs needed for the paper's experiments.
"""
from typing import List

import networkx as nx
from networkx.classes.graph import Graph
import numpy as np
import utils.synthetic_structsim as synthetic_structsim
import utils.feature_generator as feat_gen

def gen_syn1(nb_shapes=80, width_basis=300, feature_generator=None, m=5):
    """
    Synthetic Graph #1:

    Start with Barabasi-Albert graph and attach house-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)

    Returns:
        G                 :  A networkx graph
        role_id           :  A list with length equal to number of nodes in the entire graph (basis + shapes).   
                                role_id[i] is the ID of the role of node i. 
                                It is the label.
        name              :  A graph identifier
    """
    
    basis_type = 'ba'
    list_shapes = [['house']] * nb_shapes
    
    G, role_id, _ = synthetic_structsim.build_graph(width_basis, basis_type, list_shapes, start=0, m=5)
    G = perturb([G], 0.01)[0] # this function returns a graph list
    
    if feature_generator is None:
        print('passing 1 into feature generator')
        feature_generator = feat_gen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)
    
    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name

def perturb(graph_list: List[Graph], p: float):
    """ 
    Perturb the list of (sparse) graphs by adding/removing edges.
    
    Args:
        p: proportion of added edges based on current number of edges.
    
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    
    return perturbed_graph_list

def preprocess_input_graph(G: Graph, labels, normalize_adj=False):
    """ 
    Load an existing graph to be converted for the experiments.
    Args:
        G: Networkx graph to be loaded.
        labels: Associated node labels.
        normalize_adj: Should the method return a normalized adjacency matrix.
    Returns:
        A dictionary containing adjacency, node features and labels
    """
    adj = np.array(nx.to_numpy_array(G))
    
    if normalize_adj:
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=np.float32).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
        
    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]['feat'].shape[0]
    f = np.zeros((G.number_of_nodes(), feat_dim), dtype=np.float32)
    for i, u in enumerate(G.nodes()):
        f[i, :] = G.nodes[u]['feat']
    
    # add batch dim
    adj = np.expand_dims(adj, axis=0)
    f = np.expand_dims(f, axis=0)
    labels = np.expand_dims(labels, axis=0)
    
    return {'adj': adj, 'feat': f, 'labels': labels}