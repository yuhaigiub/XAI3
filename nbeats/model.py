import numpy as np

import torch
from torch import nn, Tensor

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'
    
    def __init__(self,
                 device,
                 num_nodes:int,
                 in_channels: int,
                 out_channels: int,
                 theta_dims = (4, 8),
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
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
        
        self.num_nodes = num_nodes
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
                block = block_init(num_nodes=self.num_nodes,
                                   in_channels=self.in_channels,
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
    
    def forward(self, backcast: Tensor, edge_index: Tensor = None, edge_weight: Tensor = None):
        # [batch_size, channels, num_nodes, time_steps]
        backcast = backcast.transpose(-1, -3)
        
        backcast_shape = backcast.size()
        forecast = torch.zeros(size=(*backcast_shape[: -1], self.forecast_length)).to(self.device)
        
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast, edge_index, edge_weight)

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

def linear_space_multi(num_nodes: int, backcast_length: int, forecast_length: int, is_forecast: bool):
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
                 num_nodes: int,
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
        
        self.backcast_linspace = linear_space_multi(num_nodes, backcast_length, forecast_length, is_forecast=False)
        self.forecast_linspace = linear_space_multi(num_nodes, backcast_length, forecast_length, is_forecast=True)
        
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
    
    def __batch_time_steps__(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None):
        edge_index = edge_index.expand(x.size(0), *edge_index.shape)
        
        if edge_weight is not None:
            edge_weight = edge_weight.expand(x.size(0), *edge_weight.shape)
        
        dataset = []
        for _x, _e_i, _e_w in zip(x, edge_index, edge_weight):
            dataset.append(Data(x=_x, edge_index=_e_i, edge_attr=_e_w))
        
        loader = DataLoader(dataset=dataset,  batch_size=x.size(0))
        
        return next(iter(loader))
    
    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}' \
               f'(in={self.in_channels}, hidden={self.hidden_channels}, out={self.out_channels}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'

class SeasonalityBlock(Block):
    def __init__(self,
                 num_nodes: int,
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
        super(SeasonalityBlock, self).__init__(num_nodes,
                                               in_channels,
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
            self.gcn = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None):
        x = super(SeasonalityBlock, self).forward(x)
        
        if not self.use_graph:
            x = self.feature_squeeze(x)
            # [batch_size, time_steps, num_nodes, channels]
            x = x.transpose(-1, -3)
        else:
            batch_size, channels, num_nodes, time_steps = x.size()
            
            # [batch_size, time_steps, num_nodes, channels]
            x = x.transpose(-1, -3)
            
            # [batch_size * time_steps, num_nodes, channels]
            x = x.reshape(batch_size * time_steps, num_nodes, channels)
            
            data = self.__batch_time_steps__(x, edge_index, edge_weight).to(self.device)
            gcn_output = self.gcn(data.x, data.edge_index, edge_weight=torch.flatten(data.edge_attr))
            
            # [batch_size, time_steps, num_nodes, channels]
            x = gcn_output.reshape(batch_size, time_steps, num_nodes, self.gcn.out_channels)
            
            del data, gcn_output
        
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
                 num_nodes: int,
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
        super(TrendBlock, self).__init__(num_nodes,
                                         in_channels,
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
            self.gcn = GCNConv(hidden_channels, out_channels)    
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None):
        x = super(TrendBlock, self).forward(x)
        
        if not self.use_graph:
            x = self.feature_squeeze(x)
            
            # [batch_size, time_steps, num_nodes, channels]
            x = x.transpose(-1, -3)
        else:
            batch_size, channels, num_nodes, time_steps = x.size()
            
            # [batch_size, time_steps, num_nodes, channels]
            x = x.transpose(-1, -3)
            
            # [batch_size * time_steps, num_nodes, channels]
            x = x.reshape(batch_size * time_steps, num_nodes, channels)
            
            data = self.__batch_time_steps__(x, edge_index, edge_weight).to(self.device)
            gcn_output = self.gcn(data.x, data.edge_index, edge_weight=torch.flatten(data.edge_attr))
            
            # [batch_size, time_steps, num_nodes, channels]
            x = gcn_output.reshape(batch_size, time_steps, num_nodes, self.gcn.out_channels)
            
            del data, gcn_output
        
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
                 num_nodes: int,
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
        super(GenericBlock, self).__init__(num_nodes,
                                           in_channels,
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
            self.backcast_gcn = GCNConv(hidden_channels, out_channels)
            self.forecast_gcn = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None):
        x = super(GenericBlock, self).forward(x)
        
        x = x.transpose(-3, -1)
        theta_f = self.forecast_time(x)
        
        # [batch_size, time_steps, num_nodes, channels]
        x = x.transpose(-3, -1)
        theta_f = theta_f.transpose(-3, -1)
        
        if not self.use_graph:
            theta_b = self.backcast_fc(x).transpose(-1, -3)
            theta_f = self.forecast_fc(theta_f).transpose(-1, -3)
        else:
            batch_size, channels, num_nodes, time_steps_b = x.size()
            time_steps_f = self.forecast_length
            
            # [batch_size, time_steps, num_nodes, channels]
            x = x.transpose(-1, -3)
            
            # [batch_size * time_steps, num_nodes, channels]
            x = x.reshape(batch_size * time_steps_b, num_nodes, channels)
            theta_f = theta_f.reshape(batch_size * time_steps_f, num_nodes, channels)
            
            data_b = self.__batch_time_steps__(x, edge_index, edge_weight).to(self.device)
            gcn_b = self.backcast_gcn(data_b.x, 
                                      data_b.edge_index, 
                                      edge_weight=torch.flatten(data_b.edge_attr))
            del data_b
            
            data_f = self.__batch_time_steps__(theta_f, edge_index, edge_weight).to(self.device)
            gcn_f = self.forecast_gcn(data_f.x,
                                      data_f.edge_index, 
                                      edge_weight=torch.flatten(data_f.edge_attr))
            del data_f
            
            # [batch_size, time_steps, num_nodes, channels]
            theta_b = gcn_b.reshape(batch_size, time_steps_b, num_nodes, self.backcast_gcn.out_channels)
            theta_f = gcn_f.reshape(batch_size, time_steps_f, num_nodes, self.forecast_gcn.out_channels)
            
            del gcn_b, gcn_f
            
        # [batch_size, channels, num_nodes, time_steps]
        backcast = torch.relu(theta_b).transpose(-1, -3)
        forecast = torch.relu(theta_f).transpose(-1, -3)
        
        del theta_b, theta_f
        return backcast, forecast