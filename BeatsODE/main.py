import torch

import utils
import models

def main():
    device = torch.device('cuda')
    _, _, adj = utils.load_adj('store/adj_mx.pkl')
    adj = torch.tensor(adj, dtype=torch.float32).to(device)
    
    dataloader = utils.load_dataset('store/METR-LA', 16)
    
    scaler = dataloader['scaler']
    
    model = models.BeatsODE(device, 2, 2, 12)
    model.to(device)
    
    loss_fn = utils.masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
    print('number of parameters:', len(list(model.parameters())))
    
    for epoch in range(1):
        for i, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            optimizer.zero_grad()
            
            trainx = torch.tensor(x, dtype=torch.float32).to(device)
            trainy = torch.tensor(y, dtype=torch.float32).to(device)
            _, forecast = model(trainx, adj)
            
            predict = scaler.inverse_transform(forecast)
            
            loss = loss_fn(predict, trainy, 0.0)
            loss.backward()
            
            if i % 50 == 0:
                print('epoch: {}, iter: {}, loss: {}'.format(epoch, i, round(loss.item(), 3)))
            
            optimizer.step()

if __name__ == '__main__':
    main()