"""
    graph_utils.py
    Utility for sampling graphs from a dataset.
"""
import torch

def neighborhoods(adj, n_hops):
    """Returns the n_hops degree adjacency matrix adj."""
    adj = torch.tensor(adj, dtype=torch.float32)
    hop_adj = power_adj = adj
    
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.cpu().numpy().astype(int)