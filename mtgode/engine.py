import torch
from torch import Tensor, optim

from model import MTGODE
import util

class Engine:
    def __init__(self, 
                 device,
                 in_channels: int,
                 out_channels: int,
                 conv_channels: int,
                 end_channels: int,
                 time_1: float,
                 time_step_1: float,
                 time_2: float,
                 time_step_2: float,
                 alpha: float,
                 build_A: bool,
                 predefined_A, 
                 learning_rate, wdecay, scaler):
        self.model = MTGODE(in_channels,
                            out_channels, 
                            conv_channels,
                            end_channels,
                            time_1,
                            time_step_1,
                            time_2,
                            time_step_2,
                            alpha,
                            build_A,
                            predefined_A)
        self.model.to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        
    def train(self, input: Tensor, real_val: Tensor):
        self.model.train()
        self.optimizer.zero_grad()
        
        if real_val.size() == 3:
            real = torch.unsqueeze(real_val, dim=-1)
        else:
            real = real_val
            
        predict = self.model(input)
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
        
        if real_val.size() == 3:
            real = torch.unsqueeze(real_val, dim=-1)
        else:
            real = real_val
        
        predict = self.model(input)
        if self.scaler is not None:
            predict = self.scaler.inverse_transform(predict)
        
        loss = self.loss(predict, real, 0.0)
        
        mape = util.masked_mape(predict, real, 0.0).item()
        mse = util.masked_mse(predict, real, 0.0).item()
        
        return loss.item(), mape, mse
        