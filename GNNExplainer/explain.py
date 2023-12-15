""" 
explain.py
Implementation of the explainer.
"""
import os
import time
import math
from typing import List

import torch
from torch import nn, Tensor
from torch.nn import init

import numpy as np

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import utils.graph_utils  as graph_utils
from utils.arguments import ArgumentsExplain

class ExplainerModule(nn.Module):
    def __init__(self,
                 device,
                 adj: Tensor,
                 x: Tensor,
                 model: nn.Module,
                 labels: Tensor,
                 args: ArgumentsExplain,
                 graph_idx=0,
                 writer=None,
                 use_sigmoid=True,
                 graph_mode=False):
        super(ExplainerModule, self).__init__()
        self.device = device
        self.adj = adj
        self.x = x
        self.model = model
        self.labels = labels
        
        self.mask_activation = args.mask_activation
        self.use_sigmoid = use_sigmoid
        
        self.graph_idx = graph_idx
        self.graph_mode = graph_mode
        
        self.writer = writer
        self.args = args
        
        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(num_nodes, init_strategy)
        
        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")
        
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        
        # for masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        
        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)
        
        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }
    
    def construct_edge_mask(self, num_nodes: int, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = init.calculate_gain('relu') * math.sqrt(2.0 / (num_nodes + num_nodes))
            
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            init.constant_(mask, const_val)
        
        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None
        
        return mask, mask_bias
    
    def construct_feat_mask(self, feat_dims: int, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dims))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        
        return mask
    
    def forward(self, node_idx: int, unconstrained=False, mask_features=True, marginalize=False):
        x = self.x
        
        if unconstrained:
            raise Exception("unconstrained not implemented")
        else:
            self.masked_adj = self._masked_adj()
            if mask_features:
                if self.use_sigmoid:
                    feat_mask = torch.sigmoid(self.feat_mask)
                else:
                    feat_mask = self.feat_mask
                
                if marginalize:
                    raise Exception("marginalize not implemented")
                else:
                    x = x * feat_mask
        
        ypred, adj_att = self.model(x, self.masked_adj)
        if self.graph_mode:
            raise Exception("graph mode not implemented")
        else:
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)
        
        return res, adj_att
    
    def loss(self, pred: Tensor, pred_label: Tensor, node_idx: int, epoch: int):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = True
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            if self.graph_mode:
                raise Exception("graph mode not implemented")
            else:
                pred_label_node = pred_label[node_idx]
                gt_label_node = self.labels[0][node_idx]
            gt_label_node = torch.tensor(gt_label_node, dtype=torch.int)
            logit = pred[gt_label_node]
            pred_loss = -torch.log(logit)
        
        # size
        mask = self.mask
        if self.mask_activation == 'sigmoid':
            mask = torch.sigmoid(self.mask)
        elif self.mask_activation == 'ReLU':
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs['size'] * torch.sum(mask)
        
        if self.use_sigmoid:
            feat_mask = torch.sigmoid(self.feat_mask)
        else:
            feat_mask = self.feat_mask
            
        feat_size_loss = self.coeffs['feat_size'] * torch.mean(feat_mask)
        
        # entropy
        mask_entropy = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_entropy_loss = self.coeffs['ent'] * torch.mean(mask_entropy)
        
        feat_mask_entropy = -feat_mask * torch.log(feat_mask) - (1 - feat_mask) * torch.log(1 - feat_mask)
        feeat_mask_entropy_loss = self.coeffs['feat_ent'] * torch.mean(feat_mask_entropy)
        
        # laplacian
        
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        
        if self.graph_mode:
            raise Exception("graph mode not implemented")
        else:
            m_adj = self.masked_adj[self.graph_idx]
        
        L = torch.Tensor(D - m_adj).to(self.device)
        pred_label_t = torch.tensor(pred_label, dtype=torch.float32).to(self.device)
        
        if self.graph_mode:
            raise Exception("graph mode not implemented")
        else:
            lap_loss = (self.coeffs["lap"] * (pred_label_t @ L @ pred_label_t) / self.adj.numel())
        
        loss = pred_loss + size_loss
        
        if self.writer is not None:
            raise Exception("writer not implemented")
        
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

class Explainer:
    def __init__(self,
                 device,
                 model: nn.Module,
                 adj: Tensor,
                 feat: Tensor,
                 labels: Tensor,
                 pred: Tensor,
                 train_idx: int,
                 args: ArgumentsExplain,
                 writer=None,
                 graph_mode=False,
                 graph_idx=-1,
                 print_training=True):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.labels = labels
        self.pred = pred
        
        self.train_idx = train_idx
        self.n_hops = args.num_gc_layers
        
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        
        # index of the query node in the new adj
        if graph_mode:
            self.neighborhoods = None
        else:
            self.neighborhoods = graph_utils.neighborhoods(self.adj, self.n_hops)
        
        self.args = args
        self.writer = writer
        self.print_training = print_training
        self.device = device
    
    # Main ------------------------------------------------------------------------
    def explain(self, 
                node_idx: int, 
                graph_idx=0, 
                graph_mode=False, 
                unconstrained=False, 
                model="exp"):
        """
        Explain a single node prediction
        """
        
        if graph_mode:
            raise Exception("Graph mode not implemented")
        else:
            node_idx_new, sub_adj, sub_feat, sub_labels, neighbors = self.extract_neighborhood(node_idx, graph_idx)
            
        sub_labels = np.expand_dims(sub_labels, axis=0)
        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)
        
        adj = torch.tensor(sub_adj, dtype=torch.float32).to(self.device)
        x = torch.tensor(sub_feat, dtype=torch.float32).to(self.device)
        labels = torch.tensor(sub_labels, dtype=torch.float32).to(self.device)
        
        if self.graph_mode:
            raise Exception("Graph mode not implemented")
        else:
            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
            print("Node predicted Label:", pred_label[node_idx_new])
        
        # define explainer module
        explainer = ExplainerModule(self.device,
                                    adj=adj, 
                                    x=x, 
                                    model=self.model,
                                    labels=labels,
                                    args=self.args,
                                    graph_idx=self.graph_idx,
                                    writer=self.writer,
                                    graph_mode=self.graph_mode).to(self.device)
        
        self.model.eval()
        
        if model == 'grad':
            raise Exception('model == grad not implemented')
        else:
            explainer.train()
            begin_time = time.time()
            
            for epoch in range(self.args.epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                
                ypred, adj_atts = explainer(node_idx_new, unconstrained=unconstrained)
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()
                
                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()
                
                mask_density = explainer.mask_density()
                
                if self.print_training and epoch % 10 == 0:
                    print(
                        "epoch:",        epoch, "; ",
                        "loss:",         round(loss.item(), 3), "; ",
                        "mask density:", round(mask_density.item(), 3), "; ",
                        "pred:",         ypred.detach().cpu().numpy(),
                    )
                
                single_subgraph_label = sub_labels.squeeze()
                
                if self.writer is not None:
                    raise Exception("writer is not implemented")
                
                if model != "exp":
                    break
            
            print("finished training in ", time.time() - begin_time)
            if model == "exp":
                print(explainer.masked_adj.shape, sub_adj.shape)
                masked_adj = (explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze())
            else:
                adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
                masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()
        
        fname = ('masked_adj_' + io_utils.gen_explainer_prefix(self.args) +
                 'node_idx_' + str(node_idx) + 'graph_idx_' + str(self.graph_idx) + '.npy')
        with open(os.path.join(self.args.logdir, fname), 'wb') as outfile:
            np.save(outfile, np.asarray(masked_adj.copy()))
            print("Saved adjacency matrix to ", fname)
        
        return masked_adj

    # Utils -----------------------------------------------------------------------
    def extract_neighborhood(self, node_idx: int, graph_idx=0):
        """
        Returns the neighborhood of a given node
        """
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]
        sub_labels = self.labels[graph_idx, neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_labels, neighbors
