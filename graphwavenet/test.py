import time
import os

import numpy as np

import torch

from arguments import Arguments
from engine import Engine
import util

args = Arguments()

def main():
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adj_data)
    dataloader = util.load_dataset(args.data, args.batch_size, apply_scaling=args.apply_scaling)

    scaler = dataloader['scaler'] # None if apply_scaling = False
    
    engine = Engine(args.in_dims, 
                    args.out_dims, 
                    args.num_nodes, 
                    args.seq_len, 
                    args.learning_rate, 
                    args.weight_decay, 
                    device, 
                    adj_mx, 
                    scaler)
    
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    log_file_test = open(os.path.join(args.log, 'loss_test_log.txt'), 'w')
    
    if not args.test_all: # test 1 epoch
        start = args.epoch_test
        end = args.epoch_test + 1
    else: # test multiple epochs
        start = args.last_epoch + 1
        end = args.last_epoch + 1 + args.epochs
        
    for epoch in range(start, end):
        # load model
        engine.model.load_state_dict(torch.load(args.save + "/G_T_model_" + str(epoch) + ".pth"))
        
        s1 = time.time()
        test_loss = []
        test_mape = []
        test_mse = []
        
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            
            metrics = engine.eval(testx, testy)
            
            test_loss.append(metrics[0])
            test_mape.append(metrics[1])
            test_mse.append(metrics[2])
        
        mtest_loss = np.mean(test_loss)
        mtest_mape = np.mean(test_mape)
        mtest_mse = np.mean(test_mse)
        
        log_file_test.write(f'Epoch {epoch}, Test Loss: {mtest_loss:.4f}, ' \
                            f'Test MAPE: {mtest_mape:.4f}, ' \
                            f'Test MSE: {mtest_mse:.4f} \n')
        log_file_test.flush()
    
        s2 = time.time()
        
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(epoch, (s2 - s1)))

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print('Total time spent: {:.4f}'.format(t2 - t1))
    