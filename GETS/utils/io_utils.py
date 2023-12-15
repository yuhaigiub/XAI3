import pickle
import os
import statistics
import numpy as np
import networkx as nx

from torch import Tensor

from utils.data_utils import StandardScaler, DataLoader
from utils.arguments import Arguments

import matplotlib
import matplotlib.pyplot as plt
import tensorboardX


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin')
    except Exception as e:
        print("Unable to load data:", pickle_file)
        print("Error:")
        print(e)
        raise
    
    return pickle_data

def load_adj(pickle_file):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pickle_file)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, apply_scaling=True):
    if valid_batch_size is None:
        valid_batch_size = batch_size
    if test_batch_size is None:
        test_batch_size = batch_size
        
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    
    scaler = None
    if apply_scaling:
        mean = data['x_train'][..., 0].mean()
        std = data['x_train'][..., 0].std()
        scaler = StandardScaler(mean, std)

    
    for category in ['train', 'val', 'test']:
        if apply_scaling:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    
    data['scaler'] = scaler
    
    return data

def log_matrix(writer, mat: Tensor, name: str, epoch: int, fig_size=(8, 6), dpi=200):
    """Save an image of a matrix to disk.

    Args:
        - writer    :  A file writer.
        - mat       :  The matrix to write.
        - name      :  Name of the file to save.
        - epoch     :  Epoch number.
        - fig_size  :  Size to of the figure to save.
        - dpi       :  Resolution.
    """
    
    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    mat = mat.cpu().detach().numpy()
    
    plt.imshow(mat, cmap=plt.get_cmap("BuPu"))
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor("face")
    
    plt.tight_layout()
    fig.canvas.draw()
    writer.add_image(name, tensorboardX.utils.figure_to_image(fig), epoch)

def gen_prefix(args: Arguments):
    '''
    Generate label prefix for a graph model.
    '''
    name = args.data.split('/')[-1] # METR-LA or PEMS-BAY
    
    # name += "_" + args.method
    name += "_o" + str(args.out_dim)
    
    return name

def gen_explainer_prefix(args: Arguments):
    '''
    Generate label prefix for a graph explainer model.
    '''
    name = gen_prefix(args) + "_explain"
    
    return name

def denoise_graph(adj, 
                  node_idx, 
                  feat=None, 
                  labels=None, 
                  threshold=None, 
                  threshold_num:int=None, 
                  max_component=True):
    """Cleaning a graph by thresholding its node values.

    Args:
        - adj               :  Adjacency matrix.
        - node_idx          :  Index of node to highlight (TODO ?)
        - feat              :  An array of node features.
        - label             :  A list of node labels.
        - threshold         :  The weight threshold.
        - theshold_num      :  The maximum number of nodes to threshold.
        - max_component     :  TODO
    """
    num_nodes = adj.shape[-1]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.nodes[node_idx]["self"] = 1
    
    if feat is not None:
        raise NotImplementedError
    if labels is not None:
        raise NotImplementedError
    
    if threshold_num is not None:
        adj_threshold_num = threshold_num * 2
        neighbor_size = len(adj[adj > 0])
        threshold_num = min(neighbor_size, adj_threshold_num)
        threshold = np.sort(adj[adj > 0])[-threshold_num]
    
    if threshold is not None:
        weighted_edge_list = [
            (i, j, adj[i, j])
            for i in range(num_nodes)
            for j in range(num_nodes)
            if adj[i, j] >= threshold
        ]
    else:
        weighted_edge_list = [
            (i, j, adj[i, j])
            for i in range(num_nodes)
            for j in range(num_nodes)
            if adj[i, j] > 1e-6
        ]
    G.add_weighted_edges_from(weighted_edge_list)
    
    if max_component:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    else:
        G.remove_nodes_from(list(nx.isolates(G)))
    
    return G

def log_graph(Gc,
              name,
              args: Arguments,
              epoch=0,
              fig_size=(4, 3),
              dpi=300,
              edge_vmax=None,
              identify_self=True,
              nodecolor="label"):
    cmap=plt.get_cmap("Set1")
    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    
    node_colors = []
    edge_colors = [w for (u, v, w) in Gc.edges.data("weight", default=1)]
    
    min_color = min([d for (u, v, d) in Gc.edges(data="weight", default=1)])
    
    for i in Gc.nodes():
        if identify_self and "self" in Gc.nodes[i]:
            node_colors.append(0)
        elif nodecolor == "label" and "label" in Gc.nodes[i]:
            node_colors.append(Gc.nodes[i]["label"] + 1)
        elif nodecolor == "feat" and "feat" in Gc.nodes[i]:
            raise NotImplementedError
        else:
            node_colors.append(1)
    
    if edge_vmax is None:
        edge_vmax = statistics.median_high([d for (u, v, d) in Gc.edges(data="weight", default=1)])
    
    vmax = 8
    edge_vmin = 2 * min_color - edge_vmax
    
    if Gc.number_of_nodes() == 0:
        raise Exception("empty graph")
    if Gc.number_of_edges() == 0:
        raise Exception("empty edge")
    
    pos_layout = nx.kamada_kawai_layout(Gc, weight=None)
    nx.draw(
        Gc,
        pos=pos_layout,
        with_labels=False,
        font_size=4,
        node_color=node_colors,
        vmin=0,
        vmax=vmax,
        cmap=cmap,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap("Greys"),
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        width=1.0,
        node_size=50,
        alpha=0.8)
    
    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()
    
    logdir = args.logdir
    save_path = os.path.join(logdir, name  + "_" + str(epoch) + ".pdf")
    print(logdir + "/" + name + gen_explainer_prefix(args) + "_" + str(epoch) + ".pdf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="pdf")