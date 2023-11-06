from functools import reduce
from operator import mul

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGCNConv, GCNConv

class GraphWaveNet(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 in_channels: int, 
                 out_channels: int, 
                 out_timesteps: int, 
                 dilations=[1, 2, 1, 2, 1, 2, 1, 2],
                 adaptive_embeddings=10,
                 dropout=0.3,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=512):
        super(GraphWaveNet, self).__init__()
        
        self.total_dilation = sum(dilations)
        self.num_dilations = len(dilations)
        self.num_nodes = num_nodes
        
        self.dropout = dropout
        
        self.e1 = nn.Parameter(torch.randn(num_nodes, adaptive_embeddings), requires_grad=True)
        self.e2 = nn.Parameter(torch.randn(adaptive_embeddings, num_nodes), requires_grad=True)
        
        self.input = nn.Conv2d(in_channels=in_channels, 
                               out_channels=residual_channels, 
                               kernel_size=(1, 1))
        
        self.tcn_a = nn.ModuleList()
        self.tcn_b = nn.ModuleList()
        
        self.gcn = nn.ModuleList()
        self.gcn_adp = nn.ModuleList()
        
        self.bn = nn.ModuleList()
        self.skip = nn.ModuleList()
        
        self.out_timesteps = out_timesteps
        self.out_channels = out_channels
        
        for d in dilations:
            self.tcn_a.append(nn.Conv2d(in_channels=residual_channels, 
                                        out_channels=dilation_channels, 
                                        kernel_size=(1, 2), # OG: (1, 2) 
                                        dilation=d))
            
            self.tcn_b.append(nn.Conv2d(in_channels=residual_channels,
                                        out_channels=dilation_channels,
                                        kernel_size=(1, 2), # OG: (1, 2)
                                        dilation=d))
            
            self.skip.append(nn.Conv2d(in_channels=residual_channels,
                                       out_channels=skip_channels,
                                       kernel_size=(1, 1)))
            
            # GCNConv is used for performing graph convolutions over the normal adjacency matrix
            self.gcn.append(GCNConv(in_channels=dilation_channels, out_channels=residual_channels))
            
            self.gcn_adp.append(DenseGCNConv(in_channels=dilation_channels, out_channels=residual_channels))
            
            self.bn.append(nn.BatchNorm2d(residual_channels))
            
        self.end1 = nn.Conv2d(in_channels=skip_channels, 
                              out_channels=end_channels, 
                              kernel_size=(1, 1))
        
        self.end2 = nn.Conv2d(in_channels=end_channels, 
                              out_channels=out_channels * out_timesteps, 
                              kernel_size=(1, 1))
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None):
        is_batched = True
        
        if len(x.size()) == 3:
            is_batched = False
            x = torch.unsqueeze(x, dim=0)
            
        batch_size = x.size()[:-3]
        input_timesteps = x.size(-3)
        
        # [batch_size, channels, num_nodes, time_steps]
        x = x.transpose(-1, -3)
        
        # if the dilation step require a larger sequence length, padding is applied
        if self.total_dilation + 1 > input_timesteps:
            x = F.pad(x, (self.total_dilation - input_timesteps + 1, 0))
        
        x = self.input(x)
        
        # [num_nodes, num_nodes]
        adp = F.softmax(F.relu(torch.mm(self.e1, self.e2)), dim=1)
        
        skip_out = None
        
        for k in range(self.num_dilations):
            residual = x
            
            # TCN layer
            g1 = self.tcn_a[k](x)
            g1 = torch.tanh(g1)
            
            g2 = self.tcn_b[k](x)
            g2 = torch.sigmoid(g2)
            
            g = g1 * g2
            
            # [batch_size, channels, num_nodes, time_steps]
            skip_cur = self.skip[k](x)
            
            if skip_out is None:
                skip_out = skip_cur
            else:
                # only take the required amount of latest time step to aggregate
                # because dilation shrink the number of timesteps
                skip_out = skip_out[..., -skip_cur.size(-1):] + skip_cur
            
            # [batch_size, time_steps, num_nodes, channels]
            g = g.transpose(-1, -3)
            
            time_steps = g.size(-3)
            
            # [batch_size * time_steps, num_nodes, channels]
            g = g.reshape(reduce(mul, g.size()[:-2]), *g.size()[-2:])
            
            # data.x: [batch_size * time_steps * num_nodes, channels]
            # data.edge_index: [2, batch_size * time_steps * len(edge_index)]
            # data.edge_attr: [batch_size * time_steps * len(edge_weight)]
            data = self.__batch_time_steps__(g, edge_index, edge_weight).to(g.device)
            
            gcn_output = self.gcn[k](data.x, data.edge_index, edge_weight=torch.flatten(data.edge_attr))
            gcn_output = gcn_output.reshape(*batch_size, time_steps, -1, self.gcn[k].out_channels)
            
            gcn_output_adp = self.gcn_adp[k](g, adp)
            gcn_output_adp = gcn_output_adp.reshape(*batch_size, time_steps, -1, self.gcn[k].out_channels)
            
            
            x = gcn_output + gcn_output_adp
            
            x = F.dropout(x, p=self.dropout)
            
            # [batch_size, time_steps, num_nodes, channels]
            x = x.transpose(-3, -1)
            
            x = x + residual[..., -x.size(-1):]
            
            self.bn[k](x)
        
        skip_out = skip_out[..., -1:]
        
        x = torch.relu(skip_out)
        
        # [batch_size, time_steps, num_nodes, channels]
        x = torch.relu(self.end1(x))
        x = self.end2(x)
        
        if is_batched:
            x = x.reshape(*batch_size, self.out_timesteps, self.num_nodes, self.out_channels)
        else:
            x = x.reshape(self.out_timesteps, self.num_nodes, self.out_channels)
        
        del adp, skip_out, residual, g1, g2, g, skip_cur, gcn_output, gcn_output_adp
        return x
    
    def __batch_time_steps__(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None):
        edge_index = edge_index.expand(x.size(0), *edge_index.shape)
        
        if edge_weight is not None:
            edge_weight = edge_weight.expand(x.size(0), *edge_weight.shape)
        
        dataset = []
        for _x, _e_i, _e_w in zip(x, edge_index, edge_weight):
            dataset.append(Data(x=_x, edge_index=_e_i, edge_attr=_e_w))
        
        loader = DataLoader(dataset=dataset,  batch_size=x.size(0))
        
        return next(iter(loader))