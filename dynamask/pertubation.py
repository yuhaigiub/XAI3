from abc import ABC, abstractmethod

import torch
from torch import Tensor

class Pertubation(ABC):
    @abstractmethod
    def __init__(self, device, epsilon=1.0e-7):
        self.mask = None
        self.epsilon = epsilon
        self.device = device
    
    @abstractmethod
    def apply(self, X: Tensor, mask: Tensor):
        if X is None or mask is None:
            raise NameError("Missing argument")

class FadeMovingAverage(Pertubation):
    def __init__(self, device, epsilon=1.0e-7):
        super().__init__(device, epsilon)
    
    def apply(self, x: Tensor, mask: Tensor):
        # x: [time_steps, num_nodes, channels]
        # mask: [time_steps, num_nodes]
        
        super().apply(x, mask)
        T = x.shape[0]
        
        # [time_steps, num_nodes * channels]
        avg = torch.mean(x, 0).reshape(1, -1).to(self.device)
        # [1, num_nodes, channels]
        avg = avg.reshape(1, *x.size()[1:])
        # [time_steps, num_nodes, channels]
        avg = avg.repeat(T, 1, 1)
        
        based = torch.einsum('tnc, tn -> tnc', x, mask)
        pert = torch.einsum('tnc, tn -> tnc', avg, 1 - mask)
        
        # [time_steps, num_nodes, channels]
        x_pert = based + pert
        
        return x_pert
        