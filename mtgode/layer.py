import torch
from torch import nn, Tensor

import torchdiffeq

class CGP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 alpha: float,
                 time: float,
                 step_size: float):
        super(CGP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.step_size = step_size
        self.integration_time = time
        self.estimated_nfe = round(self.integration_time / step_size)
        
        self.CGPODE: CGPODEBlock = CGPODEBlock(CGPODEFunc(in_channels, out_channels, alpha),
                                               step_size,
                                               self.estimated_nfe)
        
    def forward(self, x: Tensor, adj: Tensor):
        self.CGPODE.set_x0(x)
        self.CGPODE.set_adj(adj)
        h = self.CGPODE(x, self.integration_time)
        
        return h

class CGPODEBlock(nn.Module):
    def __init__(self, cgpfunc, step_size, estimated_nfe):
        super(CGPODEBlock, self).__init__()
        self.odefunc: CGPODEFunc = cgpfunc
        self.step_size = step_size
        self.estimated_nfe = estimated_nfe
        
        self.mlp = linear((estimated_nfe + 1) * self.odefunc.in_channels, self.odefunc.out_channels)
    
    def forward(self, x: Tensor, t):
        self.integration_time = torch.Tensor([0, t]).float().type_as(x)
        
        out = torchdiffeq.odeint(self.odefunc,
                                 x,
                                 self.integration_time,
                                 method="euler",
                                 options=dict(step_size=self.step_size))
        
        outs = self.odefunc.outs
        self.odefunc.outs = []
        outs.append(out[-1])
        h_out = torch.cat(outs, dim=1)
        h_out = self.mlp(h_out)
        
        return h_out
    
    def set_x0(self, x0: Tensor):
        self.odefunc.x0 = x0.clone().detach()
    
    def set_adj(self, adj: Tensor):
        self.odefunc.adj = adj

class CGPODEFunc(nn.Module):
    def __init__(self, in_channels, out_channels, alpha):
        super(CGPODEFunc, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.nconv = nconv()
        
        self.x0 = None
        self.adj = None
        self.outs = []
    
    def forward(self, t, x: Tensor):
        adj = self.adj + torch.eye(self.adj.size(0)).to(x.device)
        d = adj.sum(1)
        _d = torch.diag(torch.pow(d, -0.5))
        adj_norm = torch.mm(torch.mm(_d, adj), _d)
        
        self.outs.append(x)
        
        ax = self.nconv(x, adj_norm)
        f = 0.5 * self.alpha * (ax - x)
        
        return x

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()
    
    def forward(self, x: Tensor, A: Tensor):
        # x: [batch_size, channels, num_nodes, time_steps]
        # A: [num_nodes, num_nodes]
        x = torch.einsum('bcnt, wn -> bcwt', x, A)
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
    
    def forward(self, x: Tensor):
        return self.mlp(x)