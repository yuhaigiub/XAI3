import time
import math
import numpy as np

import torch
from torch import nn, Tensor

import utils.metrics as metrics
import utils.graph_utils as graph_utils
import utils.io_utils as io_utils


class ExplainerModule(nn.Module):
    def __init__(self,
                 device,
                 model: nn.Module,
                 adj: Tensor,
                 x: Tensor,
                 args,
                 graph_mode=False,
                 use_sigmoid=False,
                 writer=None):
        super(ExplainerModule, self).__init__()
        self.args = args
        self.writer = writer
        self.device = device
        
        self.model = model
        
        self.adj = adj
        self.x = x
        
        self.graph_mode = graph_mode
        self.mask_activation = args.mask_activation
        self.use_sigmoid = use_sigmoid
        
        self.num_nodes = adj.size(0)
        
        self.mask, self.mask_bias = self.construct_edge_mask(self.num_nodes, "normal")
        
        params = [self.mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        
        # for masking diagonal entries
        self.diag_mask = torch.ones(self.num_nodes, self.num_nodes) - torch.eye(self.num_nodes)
        
        self.optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        
        # hyperparameters
        self.coeffs = {
            "size": 0.005,
            "lap": 1.0,
        }
    
    def forward(self, node_idx: int, unconstrained=False):
        x = self.x
        
        if unconstrained:
            raise NotImplementedError
        else:
            self.masked_adj = self._masked_adj()
        
        # call the black box on masked input & adj
        _, ypred = self.model(x, self.masked_adj)
        
        if self.graph_mode:
            res = ypred
        else:
            res = ypred[:, :, node_idx, :]
        
        return res
    
    def construct_edge_mask(self, num_nodes: int, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        
        if init_strategy == 'normal':
            std = nn.init.calculate_gain('relu') * math.sqrt(2.0 / (num_nodes + num_nodes))
            
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == 'const':
            nn.init.constant_(mask, const_val)
        
        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None
        
        return mask, mask_bias
    
    def loss(self, pred: Tensor, pred_label: Tensor, node_idx: int, epoch: int, debug=False):
        pred_loss = metrics.masked_mae(pred, pred_label, 0.0)
        
        # size loss
        mask = self.mask
        if self.mask_activation == 'sigmoid':
            mask = torch.sigmoid(self.mask)
        elif self.mask_activation == 'ReLU':
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs['size'] * torch.sum(mask)
        
        # laplacian
        
        if debug and epoch % 10 == 0:
            print('\tpred_loss:', round(pred_loss.item(), 3), '| size_loss:', round(size_loss.item(), 3))
        
        loss = pred_loss + size_loss
        
        return loss
    
    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        
        return mask_sum / adj_sum
    
    def _masked_adj(self):
        if self.mask_activation == 'sigmoid':
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_activation == 'ReLU':
            sym_mask = nn.ReLU()(self.mask)
        else:
            sym_mask = self.mask
        
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj
        masked_adj = adj * sym_mask
        
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        
        masked_adj = masked_adj.to(self.device)
        self.diag_mask = self.diag_mask.to(self.device)
        
        return masked_adj * self.diag_mask
    
    def log_mask(self, epoch: int):
        io_utils.log_matrix(self.writer, self.mask, "mask/node_mask", epoch)
    
    def log_masked_adj(self, node_idx, epoch, name="mask/graph"):
        masked_adj = self.masked_adj.cpu().detach().numpy()
        if self.graph_mode:
            raise NotImplementedError
        else:
            G = io_utils.denoise_graph(masked_adj, node_idx, threshold_num=12, max_component=True)
            io_utils.log_graph(G, 
                               name=name,
                               args=self.args,
                               epoch=epoch)
    
    def log_adj_grad(self):
        pass

class Explainer:
    def __init__(self,
                 device,
                 model: nn.Module,
                 adj: Tensor,
                 feat: Tensor,
                 pred: Tensor,
                 args,
                 graph_mode=False,
                 print_training=True,
                 writer=None):
        self.args = args
        self.writer = writer
        
        self.device = device
        self.print_training = print_training
        
        
        self.model = model
        
        self.feat = feat
        self.adj = adj
        self.pred = pred
        
        self.n_hops = args.num_gc_layers
        self.graph_mode = graph_mode
        
        if graph_mode:
            self.neighborhoods = None
        else:
            self.neighborhoods = graph_utils.neighborhoods(self.adj, self.n_hops)
    
    def explain(self,
                node_idx: int,  
                unconstrained=False, 
                model="exp"):
        if self.graph_mode:
            node_idx_new = node_idx
            sub_adj = self.adj
            sub_feat = self.feat
            neighbors = np.asarray(range(self.adj.shape[0]))
        else:
            node_idx_new, sub_adj, sub_feat, neighbors = self.extract_neighborhood(node_idx)
        
        adj = torch.tensor(sub_adj, dtype=torch.float32).to(self.device)
        x = torch.tensor(sub_feat, dtype=torch.float32).to(self.device)
        
        
        if self.graph_mode:
            pred_label = self.pred
        else:
            pred_label = self.pred[:, :, node_idx, :]
            
        
        explainer = ExplainerModule(device=self.device,
                                    model=self.model,
                                    adj=adj,
                                    x=x,
                                    args=self.args,
                                    graph_mode=self.graph_mode, 
                                    writer=self.writer)
        explainer.to(self.device)

        self.model.eval()
        
        if model == 'grad':
            raise NotImplementedError
        else:
            explainer.train()
            begin_time = time.time()
            
            for epoch in range(self.args.epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                
                ypred = explainer(node_idx_new)
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()
                
                explainer.optimizer.step()
                
                mask_density = explainer.mask_density()
                
                if self.print_training and epoch % 10 == 0:
                    print(
                        "epoch:",        epoch, "; ",
                        "loss:",         round(loss.item(), 3), "; ",
                        "mask density:", round(mask_density.item(), 3),
                    )
                
                
                if self.writer is not None and epoch % 25 == 0:
                        # explainer.log_mask(epoch)
                        explainer.log_masked_adj(node_idx_new, epoch)
                        # explainer.log_adj_grad(node_idx_new, pred_label, epoch, label=single_subgraph_label)
            print('finished training in ', time.time() - begin_time)
            
            if model == "exp":
                masked_adj = None
            else:
                raise NotImplementedError
        
        return masked_adj
    
    def extract_neighborhood(self, node_idx: int):
        # [num_nodes,]
        neighbors_adj_row = self.neighborhoods[node_idx, :]
        
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        
        sub_adj = self.adj[neighbors][:, neighbors]
        sub_feat = self.feat[..., neighbors, :]
        
        return node_idx_new, sub_adj, sub_feat, neighbors