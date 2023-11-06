import torch
from torch import optim, Tensor

from model import GraphWaveNet
import util

class Engine:
    def __init__(self, in_channels, out_channels, num_nodes, seq_len, lrate, wdecay, device, adj_mx, scaler):
        self.model = GraphWaveNet(num_nodes, in_channels, out_channels, seq_len)
        self.model.to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        
        self.scaler = scaler
        self.clip = 5
        
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
                    
    def train(self, input: Tensor, real_val: Tensor):
        self.model.train()
        self.optimizer.zero_grad()
        
        predict = self.model(input, self.edge_index, self.edge_weight)
        
        if real_val.size() == 3:
            real = torch.unsqueeze(real_val, dim=-1)
        else:
            real = real_val
        
        if self.scaler is not None:
            predict = self.scaler.inverse_transform(predict)
        
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        
        self.optimizer.step()
        
        mape = util.masked_mape(predict, real, 0.0).item()
        mse = util.masked_mse(predict, real, 0.0).item()
        
        return loss.item(), mape, mse
    
    def eval(self, input: Tensor, real_val: Tensor):
        self.model.eval()
        
        predict = self.model(input, self.edge_index, self.edge_weight)
        
        if real_val.size() == 3:
            real = torch.unsqueeze(real_val, dim=-1)
        else:
            real = real_val
        
        if self.scaler is not None:
            predict = self.scaler.inverse_transform(predict)
        
        loss = self.loss(predict, real, 0.0)
        
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        
        return loss.item(), mape, rmse