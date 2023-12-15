import torch
from torch import optim
import numpy as np

import models

import utils.metrics as metrics
import utils.io_utils as io_utils
import utils.graph_utils as graph_utils

from utils.arguments import ArgumentsTrain

def main():
    args = ArgumentsTrain()
    device = torch.device(args.device)
    _,_, adj_mx = io_utils.load_adj(args.adj_data)
    # adj_mx = np.where(adj_mx > 0, 1.0, 0.0)
    
    dataloader = io_utils.load_dataset(args.data, args.batch_size)
    scaler = dataloader['scaler']
    
    # model = models.NBeatsNet(device, args.in_dim, args.out_dim)
    model = models.GraphWaveNet()
    model.to(device)
    
    loss_fn = metrics.masked_mae
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    edge_index, edge_weight = graph_utils.process_adj(adj_mx, args.num_nodes)
    adj_mx = torch.tensor(adj_mx, dtype=torch.float32).to(device)
    edge_index.to(device)
    edge_weight.to(device)
    
    model.train()
    for epoch in range(args.epochs):
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            
            optimizer.zero_grad()
            _, forecast = model(trainx, adj_mx)
            
            if scaler is not None:
                predict = scaler.inverse_transform(forecast)
            
            loss = loss_fn(predict, trainy, 0.0)
            loss.backward()
            
            optimizer.step()
            
            if iter % 100 == 0:
                print('epoch:', epoch, ' iter:', iter, ' - ', loss.item())
    
    # save information to use in explainer
    # save model
    torch.save(model.state_dict(), args.save + "/G_T_model" + ".pth")

if __name__ == '__main__':
    main()
    print('-----done-----')