import time
import os

import numpy as np

import torch
from torch import nn, Tensor, optim

from arguments import Arguments
import util
import blackbox
from pertubation import FadeMovingAverage

args = Arguments()
MAX_ITER = 500  # TODO: temporary


def main():
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adj_data)
    dataloader = util.load_dataset(args.data, args.batch_size, apply_scaling=args.apply_scaling)
    scaler = dataloader['scaler']
    
    permutation = np.random.permutation(len(dataloader['x_train']))
    shuffled_trainx = dataloader['x_train'][permutation]
    samples = shuffled_trainx[:args.samples]
    
    # load blackbox
    model = blackbox.NBeatsNet(device,
                               args.num_nodes,
                               args.in_dims,
                               args.out_dims,
                               args.theta_dims,
                               args.stack_types,
                               args.number_of_blocks_per_stack,
                               args.forecast_length,
                               args.backcast_length,
                               args.use_graph)
    model.to(device)
    model.load_state_dict(torch.load(args.black_box_file))
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    
    pertubate = FadeMovingAverage(device)
    
    def f(x: Tensor):
        # x : [time_steps, num_nodes, channels]
        backcast, forecast = model(x.unsqueeze(dim=0))
        y: Tensor = forecast.squeeze(dim=0)
        y: Tensor = scaler.inverse_transform(y)
        
        return y
    
    print('start training...', flush=True)
    
    time_steps = args.forecast_length
    num_nodes = args.num_nodes
    
    for i, sample in enumerate(samples):
        print(f'training sample {i}\n' + '=' * 50)
        keep_ratio = args.keep_ratio
        size_reg_factor = args.size_reg_factor_init
        time_reg_factor = args.time_reg_factor
        reg_multiplicator = np.exp(np.log(args.size_reg_factor_dilation) / args.sample_iters)
        
        x = torch.Tensor(sample).to(device)
        y = f(x)
        
        # [time_steps, num_nodes]
        initial_mask = args.initial_mask_coeff * torch.ones(size=x.shape[:-1], device=device)
        mask = initial_mask.clone().detach().requires_grad_(True)
        
        optimizer = optim.Adam([mask], lr=args.learning_rate, weight_decay=args.weight_decay)
        loss_fn = util.masked_mae

        reg_ref = torch.zeros(int((1 - keep_ratio) * time_steps * num_nodes))
        reg_ref = torch.cat((reg_ref, torch.ones(time_steps * num_nodes - reg_ref.shape[0]))).to(device)

        errors = []
        size_regs = []
        time_regs = []
        
        for iter in range(args.sample_iters):
            optimizer.zero_grad()
            xm = pertubate.apply(x, mask)
            ym = f(xm)
            
            error = loss_fn(ym, y, 0.0)
            
            # regularization
            mask_sorted = mask.reshape(time_steps * num_nodes).sort()[0]
            size_reg: Tensor = ((reg_ref - mask_sorted) ** 2).mean()
            time_reg: Tensor = (torch.abs(mask[1: time_steps - 1, :] - mask[: time_steps - 2, :])).mean()
            
            loss = error + size_reg_factor * size_reg + time_reg_factor * time_reg
            loss.backward(retain_graph=True)
            
            optimizer.step()
            
            errors.append(error.item())
            size_regs.append(size_reg.detach().cpu())
            time_regs.append(time_reg.detach().cpu())
            
            mask.data = mask.data.clamp(0, 1)
            size_reg_factor = size_reg_factor * reg_multiplicator
            
            
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train MAE: {:.4f}, size_reg: ' + '{:.4f}, time_reg: {:.4f}'
                print(log.format(iter, errors[-1], size_regs[-1], time_regs[-1]), flush=True)
        
        filename = args.save + '/saliency_' + str(i) + '.npz'
        # with open(filename, 'w') as f:
        np.savez(filename, mask.detach().cpu().numpy())
        
        break

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print('Total time spent: {:.4f}'.format(t2 - t1))
