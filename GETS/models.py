import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import init

from torch_geometric.nn import GCNConv

class GraphConv(nn.Module):
    def __init__(self,
                 device,
                 in_dim,
                 out_dim,
                 dropout=0.0,
                 bias=True,
                 normalize_embedding=True):
        super(GraphConv, self).__init__()
        self.device = device
        self.dropout = dropout
        
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        
        self.normalize_embedding = normalize_embedding
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.bias = None
    
    def forward(self, x: Tensor, adj: Tensor):
        '''
        x: [batch_size, time_steps, num_nodes, channels]
        adj: [num_nodes, num_nodes]
        '''
        batch_size, time_steps, num_nodes = x.size()[:-1]
        x = x.reshape(batch_size * time_steps, *x.size()[-2:])
        
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        
        y = torch.matmul(adj, x)
        
        y = torch.matmul(y, self.weight)
        
        if self.bias is not None:
            y = y + self.bias
        
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=1)
        
        y = y.reshape(batch_size, time_steps, num_nodes, self.out_dim)
        
        return y

class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'
    
    def __init__(self,
                 device,
                 in_channels: int,
                 out_channels: int,
                 theta_dims = (4, 8, 4),
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK, GENERIC_BLOCK),
                 number_of_blocks_per_stack=3,
                 forecast_length=12,
                 backcast_length=12,
                 use_graph=True,
                 share_weights_in_stack=False,
                 hidden_channels: int = 64,
                 nb_harmonics=None):
        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        self.theta_dims = theta_dims

        self.number_of_blocks_per_stack = number_of_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        
        self.stacks = nn.ModuleList()
        self.device = device
        self.use_graph = use_graph
        
        print('| N-Beats', '(use_graph)' if use_graph else '')
        
        for stack_id in range(len(stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []
    
    def create_stack(self, stack_id: int):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type} (#{stack_id}) (share_weights={self.share_weights_in_stack})')
        
        blocks = nn.ModuleList()
        for block_id in range(self.number_of_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]
            else:
                block = block_init(in_channels=self.in_channels,
                                   hidden_channels=self.hidden_channels,
                                   out_channels=self.out_channels,
                                   theta_dim=self.theta_dims[stack_id],
                                   device=self.device,
                                   backcast_length=self.backcast_length,
                                   forecast_length=self.forecast_length,
                                   share_thetas=self.share_weights_in_stack,
                                   nb_harmonics=self.nb_harmonics,
                                   use_graph=self.use_graph)

            print(f'     | -- {block}')
            blocks.append(block)
            
        return blocks
    
    @staticmethod
    def select_block(block_type: str):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock
    
    def forward(self, backcast: Tensor, adj: Tensor):
        # [batch_size, channels, num_nodes, time_steps]
        backcast = backcast.transpose(-1, -3)
        
        backcast_shape = backcast.size()
        forecast = torch.zeros(size=(*backcast_shape[: -1], self.forecast_length)).to(self.device)
        
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast, adj)

                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
                
                block_type = self.stacks[stack_id][block_id].__class__.__name__
                layer_name = f'stack_{stack_id}-{block_type}_{block_id}'
                
                if self._gen_intermediate_outputs:
                    self._intermediary_outputs.append({'value': f.detach().numpy(), 'layer': layer_name})
                
        # [batch_size, time_steps, num_nodes, channels]
        backcast = backcast.transpose(-3, -1)
        forecast = forecast.transpose(-3, -1)
        
        return backcast, forecast
    
    def get_generic_and_interpretable_outputs(self):
        g_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' in a['layer'].lower()])
        i_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' not in a['layer'].lower()])
        
        outputs = {o['layer']: o['value'][0] for o in self._intermediary_outputs}
        
        return g_pred, i_pred, outputs
    
    def disable_intermediate_outputs(self):
        self._gen_intermediate_outputs = False
    
    def enable_intermediate_outputs(self):
        self._gen_intermediate_outputs = True

def squeeze_last_dim(tensor: Tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:
        return tensor[..., 0]
    return tensor

def linear_space(backcast_length: int, forecast_length: int, is_forecast: bool):
    horizon = forecast_length if is_forecast else backcast_length
    
    return np.arange(0, horizon) / horizon

def linear_space_multi(backcast_length: int, forecast_length: int, is_forecast: bool):
    horizon = forecast_length if is_forecast else backcast_length
    t = np.arange(0, horizon) / horizon # [time_steps]
    # t = np.repeat([t], num_nodes, axis=0) # [num_nodes, time_steps]
    return t

def seasonality_model(thetas: Tensor, t: Tensor, device):
    p = thetas.size()[-1] # theta_dim
    assert p <= thetas.shape[-1], 'theta_dim is too big'
    
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    
    s1 = torch.Tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float() # H/2 - 1
    s2 = torch.Tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    
    return torch.einsum('abcd, dy -> abcy',thetas, S.to(device))

def trend_model(thetas: Tensor, t: Tensor, device):
    p = thetas.size()[-1] # theta_dim
    assert p <= 4, 'theta_dim is too big'
    
    T = torch.tensor(np.array([t ** i for i in range(p)])).float()
    
    return torch.einsum('abcd, dy -> abcy', thetas, T.to(device))

class Block(nn.Module):
    def __init__(self,
                 in_channels: int, 
                 hidden_channels: int,
                 out_channels: int,
                 theta_dim: int,
                 device, 
                 backcast_length: int, 
                 forecast_length: int, 
                 share_thetas: bool=False, 
                 use_graph: bool=True,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.use_graph = use_graph
        
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1))
        self.fc2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1))
        self.fc3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1))
        self.fc4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1))
        
        self.device = device
        
        self.backcast_linspace = linear_space_multi(backcast_length, forecast_length, is_forecast=False)
        self.forecast_linspace = linear_space_multi(backcast_length, forecast_length, is_forecast=True)
        
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Conv2d(backcast_length, 
                                                          theta_dim,
                                                          kernel_size=(1, 1), 
                                                          bias=False)
        else:
            self.theta_f_fc = nn.Conv2d(backcast_length, theta_dim, kernel_size=(1, 1), bias=False)
            self.theta_b_fc = nn.Conv2d(backcast_length, theta_dim, kernel_size=(1, 1), bias=False)
    
    def forward(self, x: Tensor):
        x = x.to(self.device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        
        # [batch_size, channels (hidden), num_nodes, time_steps]
        return x
    
    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}' \
               f'(in={self.in_channels}, hidden={self.hidden_channels}, out={self.out_channels}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'

class SeasonalityBlock(Block):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 theta_dim: int,
                 device,
                 backcast_length: int,
                 forecast_length: int,
                 nb_harmonics=None,
                 share_thetas: bool=False,
                 use_graph: bool=True):
        super(SeasonalityBlock, self).__init__(in_channels,
                                               hidden_channels,
                                               out_channels,
                                               theta_dim,
                                               device,
                                               backcast_length,
                                               forecast_length,
                                               share_thetas,
                                               use_graph)
        if not use_graph:
            self.feature_squeeze = nn.Conv2d(hidden_channels, out_channels, kernel_size=(1, 1))
        else:
            self.gcn = GraphConv(self.device, hidden_channels, out_channels)
        
        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=init.calculate_gain('relu'))
                
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
    
    def forward(self, x: Tensor, adj: Tensor):
        x = super(SeasonalityBlock, self).forward(x)
        
        if not self.use_graph:
            x = self.feature_squeeze(x)
            # [batch_size, time_steps, num_nodes, channels]
            x = x.transpose(-1, -3)
        else:
            # [batch_size, time_steps, num_nodes, channels]
            x = x.transpose(-1, -3)
            x = self.gcn(x, adj)
        
        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)
        
        # [batch_size, channels, num_nodes, time_steps]
        theta_b = theta_b.transpose(-3, -1)
        theta_f = theta_f.transpose(-3, -1)
        
        backcast = seasonality_model(theta_b, self.backcast_linspace, self.device)
        forecast = seasonality_model(theta_f, self.forecast_linspace, self.device)
        
        del theta_b, theta_f
        return backcast, forecast

class TrendBlock(Block):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 theta_dim: int,
                 device,
                 backcast_length: int,
                 forecast_length: int,
                 nb_harmonics=None,
                 share_thetas: bool=False,
                 use_graph: bool=True):
        super(TrendBlock, self).__init__(in_channels,
                                         hidden_channels,
                                         out_channels,
                                         theta_dim,
                                         device,
                                         backcast_length,
                                         forecast_length,
                                         share_thetas, 
                                         use_graph)
        if not use_graph:
            self.feature_squeeze = nn.Conv2d(hidden_channels, out_channels, kernel_size=(1, 1))
        else:
            self.gcn = GraphConv(self.device, hidden_channels, out_channels)    
        
        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=init.calculate_gain('relu'))
                
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
    
    def forward(self, x: Tensor, adj: Tensor):
        x = super(TrendBlock, self).forward(x)
        
        if not self.use_graph:
            x = self.feature_squeeze(x)
            
            # [batch_size, time_steps, num_nodes, channels]
            x = x.transpose(-1, -3)
        else:
            # [batch_size, time_steps, num_nodes, channels]
            x = x.transpose(-1, -3)
            x = self.gcn(x, adj)
        
        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)
        
        # [batch_size, channels, num_nodes, time_steps]
        theta_b = theta_b.transpose(-3, -1)
        theta_f = theta_f.transpose(-3, -1)
        
        backcast = trend_model(theta_b, self.backcast_linspace, self.device)
        forecast = trend_model(theta_f, self.forecast_linspace, self.device)
        
        del theta_b, theta_f
        
        return backcast, forecast

class GenericBlock(Block):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 theta_dim: int,
                 device,
                 backcast_length: int,
                 forecast_length: int,
                 nb_harmonics=None,
                 share_thetas=False, 
                 use_graph=True):
        super(GenericBlock, self).__init__(in_channels,
                                           hidden_channels,
                                           out_channels,
                                           theta_dim,
                                           device,
                                           backcast_length,
                                           forecast_length,
                                           share_thetas, 
                                           use_graph)
        
        self.forecast_time = nn.Conv2d(backcast_length, forecast_length, kernel_size=(1, 1))
        
        if not use_graph:
            self.backcast_fc = nn.Conv2d(hidden_channels, out_channels, kernel_size=(1, 1))
            self.forecast_fc = nn.Conv2d(hidden_channels, out_channels, kernel_size=(1, 1))
        else:
            self.backcast_gcn = GraphConv(self.device, hidden_channels, out_channels)
            self.forecast_gcn = GraphConv(self.device, hidden_channels, out_channels)
        
        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=init.calculate_gain('relu'))
                
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
    
    def forward(self, x: Tensor, adj: Tensor):
        x = super(GenericBlock, self).forward(x)
        
        # [batch_size, time_steps, num_nodes, channels]
        x = x.transpose(-3, -1)
        theta_f = self.forecast_time(x)
        
        if not self.use_graph:
            theta_b = self.backcast_fc(x.transpose(-1, -3))
            theta_f = self.forecast_fc(theta_f.transpose(-1, -3))
        else:
            theta_b = self.backcast_gcn(x, adj).transpose(-1, -3)
            theta_f = self.forecast_gcn(theta_f, adj).transpose(-1, -3)
        
        # [batch_size, channels, num_nodes, time_steps]
        backcast = torch.relu(theta_b)
        forecast = torch.relu(theta_f)
        
        del theta_b, theta_f
        return backcast, forecast

class GraphWaveNet(nn.Module):
    def __init__(self,
                #  num_nodes: int,
                 in_channels: int, 
                 out_channels: int, 
                 out_timesteps: int, 
                 dilations=[1, 2, 1, 2, 1, 2, 1, 2],
                 dropout=0.3,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=512):
        super(GraphWaveNet, self).__init__()
        
        self.total_dilation = sum(dilations)
        self.num_dilations = len(dilations)
        # self.num_nodes = num_nodes
        
        self.dropout = dropout
        
        # self.e1 = nn.Parameter(torch.randn(num_nodes, adaptive_embeddings), requires_grad=True)
        # self.e2 = nn.Parameter(torch.randn(adaptive_embeddings, num_nodes), requires_grad=True)
        
        self.input = nn.Conv2d(in_channels=in_channels,
                               out_channels=residual_channels, 
                               kernel_size=(1, 1))
        
        self.tcn_a = nn.ModuleList()
        self.tcn_b = nn.ModuleList()
        
        self.gcn = nn.ModuleList()
        # self.gcn_adp = nn.ModuleList()
        
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
            
            # self.gcn_adp.append(DenseGCNConv(in_channels=dilation_channels, 
            #                                  out_channels=residual_channels))
            
            self.bn.append(nn.BatchNorm2d(residual_channels))
            
        self.end1 = nn.Conv2d(in_channels=skip_channels, 
                              out_channels=end_channels, 
                              kernel_size=(1, 1))
        
        self.end2 = nn.Conv2d(in_channels=end_channels, 
                              out_channels=out_channels * out_timesteps,
                              kernel_size=(1, 1))
    
    def forward(self, x: Tensor, adj: Tensor):
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
            
            # time_steps = g.size(-3)
            
            gcn_output = self.gcn[k](x, adj)
            
            # gcn_output_adp = self.gcn_adp[k](g, adp)
            # gcn_output_adp = gcn_output_adp.reshape(*batch_size, time_steps, -1, self.gcn[k].out_channels)
            
            # x = gcn_output + gcn_output_adp
            
            x = gcn_output
            
            x = F.dropout(x, p=self.dropout)
            
            # [batch_size, channels, num_nodes, time_steps]
            x = x.transpose(-3, -1)
            
            print(x.shape)
            raise Exception('break')
            
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