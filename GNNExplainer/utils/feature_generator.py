""" 
featgen.py
Node feature generators.
"""

import networkx as nx
from networkx.classes.graph import Graph

import numpy as np

import abc

class FeatureGen(metaclass=abc.ABCMeta):
    """Feature Generator base class."""
    @abc.abstractmethod
    def gen_node_features(self, G):
        pass

class ConstFeatureGen(FeatureGen):
    """Constant Feature class."""
    def __init__(self, val):
        self.val = val
    
    def gen_node_features(self, G: Graph):
        feat_dict = {}
        for i in G.nodes():
            feat_dict[i] = {'feat': np.array(self.val, dtype=np.float32)}
            
        nx.set_node_attributes(G, feat_dict)
        
        # print ('feat_dict[0]["feat"]:', feat_dict[0]['feat'].dtype)
        # print ('G.nodes[0]["feat"]:', G.nodes[0]['feat'].dtype)