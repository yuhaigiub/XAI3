import torch

def process_adj(adj_mx, num_nodes):
    edge_index = [[], []]
    edge_weight = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mx.item((i, j)) != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_weight.append(adj_mx.item(i, j))
    
    edge_index = torch.tensor(edge_index, dtype=torch.int)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    
    return edge_index, edge_weight

def neighborhoods(adj, n_hops):
    # [num_nodes, num_nodes]
    adj = torch.tensor(adj, dtype=torch.float32)
    hop_adj = power_adj = adj
    
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.cpu().numpy().astype(int)