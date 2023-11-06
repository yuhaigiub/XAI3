import torch
from torch import optim, Tensor

from model import NBeatsNet
import util

class Engine:
    def __init__(self, 
                 device, 
                 num_nodes: int, 
                 in_channels: int,
                 out_channels: int,
                 forecast_length: int,
                 backcast_length: int,
                 stack_types,
                 theta_dims,
                 number_of_blocks_per_stack: int,
                 learning_rate: float, 
                 wdecay: float, 
                 scaler, 
                 use_graph: bool,
                 adj_mx = None):
        self.model = NBeatsNet(device,
                                num_nodes,
                                in_channels,
                                out_channels,
                                theta_dims,
                                stack_types,
                                number_of_blocks_per_stack,
                                forecast_length,
                                backcast_length,
                                use_graph)
        self.model.to(device)
        
        self.loss = util.masked_mae
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=wdecay)
        self.scaler = scaler
        
        self.use_graph = use_graph
        
        if use_graph:
            edge_index =[[], []]
            edge_weight = []
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj_mx.item((i, j)) != 0:
                        edge_index[0].append(i)
                        edge_index[1].append(j)
                        edge_weight.append(adj_mx.item(i, j))
            
            self.edge_index: Tensor = torch.Tensor(edge_index).type(torch.int32)
            self.edge_weight: Tensor = torch.Tensor(edge_weight)
        
        print('parameters():', len(list(self.model.parameters())))
        
    def train(self, input: Tensor, real_val: Tensor):
        self.model.train()
        self.optimizer.zero_grad()
        
        if real_val.size() == 3:
            real = torch.unsqueeze(real_val, dim=-1)
        else:
            real = real_val
        
        if self.use_graph:
            backcast, forecast = self.model(input, self.edge_index, self.edge_weight)
        else:
            backcast, forecast = self.model(input)
            
        if self.scaler is not None:
            predict = self.scaler.inverse_transform(forecast)
        else:
            predict = forecast
        
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        
        self.optimizer.step()
        
        mape = util.masked_mape(predict, real, 0.0).item()
        mse = util.masked_mse(predict, real, 0.0).item()
        
        return loss.item(), mape, mse
    
    def eval(self, input: Tensor, real_val: Tensor):
        self.model.eval()
        
        if real_val.size() == 3:
            real = torch.unsqueeze(real_val, dim=-1)
        else:
            real = real_val
            
        if self.use_graph:
            backcast, forecast = self.model(input, self.edge_index, self.edge_weight)
        else:
            backcast, forecast = self.model(input)
        
        if self.scaler is not None:
            predict = self.scaler.inverse_transform(forecast)
        else:
            predict = forecast
        
        loss = self.loss(predict, real, 0.0)
        
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        
        return loss.item(), mape, rmse