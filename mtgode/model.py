import torch
from torch import nn, Tensor
import torch.nn.functional as F

import torchdiffeq

from layer import CGP

class MTGODE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_channels: int,
                 end_channels: int,
                 time_1: float,
                 step_size_1: float,
                 time_2: float,
                 step_size_2: float,
                 alpha: float,
                 build_A: bool,
                 predefined_A: Tensor):
        super(MTGODE, self).__init__()
        self.build_A = build_A
        self.predefined_A = predefined_A
        
        self.integration_time = time_1
        self.step_size = step_size_1
        
        self.ODE = ODEBlock(ODEFunc(STBlock(conv_channels, alpha, time_2, step_size_2)),
                            step_size_1)
        
        self.start_conv = nn.Conv2d(in_channels, conv_channels, kernel_size=(1, 1))
        self.end_conv_0 = nn.Conv2d(conv_channels, end_channels // 2, kernel_size=(1, 1))
        self.end_conv_1 = nn.Conv2d(end_channels // 2, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, out_channels, kernel_size=(1, 1))
        
    def forward(self, x: Tensor):
        # [batch_size, channels, num_nodes, time_steps]
        x = x.transpose(-1, -3)
        
        adp = self.predefined_A
        
        x = self.start_conv(x)

        self.ODE.odefunc.stnet.set_graph(adp)
        x = self.ODE(x, self.integration_time)
        
        # x = x[..., -1:] # single step
        x = F.layer_norm(x, tuple(x.shape[1:]), weight=None, bias=None, eps=1e-5)
        
        x = F.relu(self.end_conv_0(x))
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        
        # [batch_size, time_steps, num_nodes, channels]
        x = x.transpose(-1, -3)
        
        return x
        
class ODEBlock(nn.Module):
    def __init__(self, odefunc, step_size):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.step_size = step_size
        
    def forward(self, x: Tensor, t):
        self.integration_time = torch.Tensor([0, t]).float().type_as(x)
        
        out = torchdiffeq.odeint(self.odefunc,
                                 x,
                                 self.integration_time,
                                 method="euler",
                                 options=dict(step_size=self.step_size))
        
        return out[-1]

class ODEFunc(nn.Module):
    def __init__(self, stnet):
        super(ODEFunc, self).__init__()
        self.stnet = stnet
    
    def forward(self, t, x: Tensor):
        return self.stnet(x)

class STBlock(nn.Module):
    def __init__(self, 
                 hidden_channels: int,
                 alpha: float,
                 time: float,
                 step_size: float,
                 dropout: float = 0.3):
        super(STBlock, self).__init__()
        self.dropout = dropout
        self.graph: Tensor | None = None
        
        self.gconv_1 = CGP(hidden_channels, hidden_channels, alpha, time, step_size)
        self.gconv_2 = CGP(hidden_channels, hidden_channels, alpha, time, step_size)
    
    def forward(self, x: Tensor):
        x = F.dropout(x, self.dropout)
        
        x = self.gconv_1(x, self.graph) + self.gconv_2(x, self.graph.T)
        return x
    
    def set_graph(self, graph: Tensor):
        self.graph = graph