"""
synthetic_structsim.py
Utilities for generating certain graph shapes.
"""
from typing import List

import networkx as nx
import numpy as np
import math

def build_graph(width_basis: int, 
                basis_type: str, 
                list_shapes: List[List[str]], 
                start=0,
                rdm_basis_plugins=False,
                add_random_edges=0,
                m=5):
    """
    This function creates a basis (scale-free, path, or cycle) 
        and attaches elements of the type in the list randomly along the basis.
    Possibility to add random edges afterwards.
    
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape,
                            next args:args for building the shape,
                            except for the start)
    start            :      initial nb for the first node
    rdm_basis_plugins:      boolean. Should the shapes be randomly placed
                            along the basis (True) or regularly (False)?
    add_random_edges :      nb of edges to randomly add on the structure
    m                :      number of edges to attach to existing node (for BA graph)
    
    OUTPUT:
    --------------------------------------------------------------------------------------
    basis            :      a nx graph with the particular shape
    role_ids         :      labels for each role
    plugins          :      node ids with the attached shapes
    """
    
    if basis_type == 'ba':
        basis, role_id = eval(basis_type)(start, width_basis, m=m)
    else:
        basis, role_id = eval(basis_type)(start, width_basis)
    
    n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    start += n_basis # indicator of the id of the next node
    
    # sampling (with replacement) where to attach the new motifs
    if rdm_basis_plugins:
        plugins = np.random.choice(n_basis, n_shapes, replace=True)
    else:
        spacing = math.floor(n_basis / n_shapes)
        plugins = [int(k * spacing) for k in range(n_shapes)]
    seen_shapes = {"basis": [0, n_basis]}
    
    for shape_id, shape in enumerate(list_shapes):
        shape_type = shape[0]
        args = [start]
        
        if len(shape) > 1:
            args += shape[1:]
        args += [0]
        
        graph_s, role_graph_s = eval(shape_type)(*args)
        n_s = nx.number_of_nodes(graph_s)
        
        try:
            col_start = seen_shapes[shape_type][0]
        except:
            col_start = np.max(role_id) + 1
            seen_shapes[shape_type] = [col_start, n_s]
        
        # Attach the shape to the basis
        basis.add_nodes_from(graph_s.nodes())
        basis.add_edges_from(graph_s.edges())
        basis.add_edges_from([(start, plugins[shape_id])])
        
        if shape_type == 'cycle':
            raise Exception("Cycle not implemented")
        
        temp_labels = [r + col_start for r in role_graph_s]
        role_id += temp_labels
        start += n_s
    
    if add_random_edges > 0:
        raise Exception("Add Random Edges not implemented")
        
    return basis, role_id, plugins

def ba(start: int, width: int, role_start=0, m=5):
    """
    Builds a BA preferential attachment graph, with index of nodes starting at start and role_ids at role_start
    
    INPUT:
    -------------
    start       :    starting index for the shape
    width       :    int size of the graph
    role_start  :    starting index for the roles
    
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at role_start)
    """
    
    graph = nx.barabasi_albert_graph(width, m)
    graph.add_nodes_from(range(start, start + width))
    
    nids = sorted(graph)
    mapping = {nid: start + i for i, nid in enumerate(nids)}
    
    graph = nx.relabel_nodes(graph, mapping)
    roles = [role_start for _ in range(width)]
    
    return graph, roles

def house(start: int, role_start=0):
    """
    Builds a house-like graph, with index of nodes starting at start and role_ids at role_start
    
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at role_start)
    """
    
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 5))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start + 1, role_start + 1, role_start + 2]
    
    return graph, roles