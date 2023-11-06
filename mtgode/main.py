import time
import os

import numpy as np

import torch

from engine import Engine
from arguments import Arguments
import util

args = Arguments()
MAX_ITER = 500 # TODO: temporary

def main():
    device = torch.device(args.device)
    dataloader = util.load_dataset(args.data, args.batch_size, apply_scaling=args.apply_scaling)
    
    scaler = dataloader['scaler']
    
    _, _, adj_mx = util.load_adj(args.adj_data)
    predefined_A = torch.Tensor(adj_mx) - torch.eye(args.num_nodes) # remove self-loop
    predefined_A = predefined_A.to(device)
    
    engine = Engine(device,
                    args.in_dims,
                    args.out_dims,
                    args.conv_channels,
                    args.end_channels,
                    args.time_1,
                    args.time_step_1,
                    args.time_2,
                    args.time_step_2,
                    args.alpha,
                    args.build_A,
                    predefined_A,
                    
                    args.learning_rate,
                    args.weight_decay,
                    scaler)
    
    if args.last_epoch != -1:
        engine.model.load_state_dict(torch.load(args.save + "/G_T_model_" + str(args.last_epoch) + ".pth"))
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    
    log_file_train = open(os.path.join(args.log, 'loss_train_log.txt'), 'w')
    log_file_val = open(os.path.join(args.log, 'loss_val_log.txt'), 'w')
    
    print('start training...', flush=True)
    
    train_time = []
    val_time = []
    his_loss = []
    
    best_epoch = 0
    
    # train + val ********************************************
    # from 0 to <epochs - 1> (with offset = args.last_epoch + 1)
    for i in range(args.last_epoch + 1, args.last_epoch + 1 + args.epochs):
        print(f'training epoch {i}\n' + '=' * 50)
        train_loss = []
        train_mape = []
        train_mse = []
        
        t1 = time.time()
        
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            # TODO: temporary
            if iter > MAX_ITER:
                break
            
            # [batch_size, time_steps, num_nodes, channels]
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            
            metrics = engine.train(trainx, trainy[:, :, :, :])
            
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_mse.append(metrics[2])
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: ' + '{:.4f}, Train MSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_mse[-1]), flush=True)
        
        t2 = time.time()
        train_time.append(t2 - t1)
        
        # save model
        torch.save(engine.model.state_dict(), args.save + "/G_T_model_" + str(i) + ".pth")
        
        if not args.validate_result:
            continue
        
        valid_loss = []
        valid_mape = []
        valid_mse = []
        
        s1 = time.time()
        print(f'validating epoch {i}\n' + '-' * 50)
        
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            # TODO: temporary
            if iter > MAX_ITER:
                break
            
            valx = torch.Tensor(x).to(device)
            valy = torch.Tensor(y).to(device)
            
            metrics = engine.eval(valx, valy[:, :, :, :])
            
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_mse.append(metrics[2])
            
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        
        val_time.append(s2 - s1)
        
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_mse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_mse)
        
        his_loss.append(mvalid_loss)
        
        if np.argmin(his_loss) == len(his_loss) - 1:
            best_epoch = i
        
        log = 'Epoch: {:03d}, '+ \
              'Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, ' + \
              'Val Loss: {:.4f}, Val MAPE: {:.4f}, Val RMSE: {:.4f}, ' + \
              'Training Time: {:.4f}/epoch'
        
        print(log.format(i,
                         mtrain_loss, 
                         mtrain_mape, 
                         mtrain_rmse, 
                         mvalid_loss,
                         mvalid_mape,
                         mvalid_rmse, 
                         (t2 - t1)))
        
        # write to log files
        log_file_train.write(f'Epoch {i}, Train Loss: {mtrain_loss:.4f}, ' \
                             f'Train MAPE: {mtrain_mape:.4f}, ' \
                             f'Train RMSE: {mtrain_rmse:.4f} \n')
        log_file_train.flush()
        
        log_file_val.write(f'Epoch {i}, Val Loss: {mvalid_loss:.4f}, '\
                           f'Val MAPE: {mvalid_mape:.4f}, ' \
                           f'Val RMSE: {mvalid_rmse:.4f} \n')
        log_file_val.flush()
        
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print(f"Best epoch recorded: {best_epoch}")
if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print('Total time spent: {:.4f}'.format(t2 - t1))