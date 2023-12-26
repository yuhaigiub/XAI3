import warnings
warnings.filterwarnings('ignore')

import sys
import os
import shutil
from tensorboardX import SummaryWriter

import numpy as np
import torch

import models
from explain import Explainer

import utils.metrics as metrics
import utils.io_utils as io_utils
from utils.arguments import ArgumentsExplain

def main():
    args = ArgumentsExplain()
    device = torch.device(args.device)
    
    # Configure the logging directory 
    if args.writer:
        path = os.path.join(args.logdir, io_utils.gen_explainer_prefix(args))
        if os.path.isdir(path) and args.clean_log:
           print('Removing existing log dir: ', path)
           if not input("Are you sure you want to remove this directory? (y/n): ").lower().strip()[:1] == "y": 
               sys.exit(1)
           shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None
    
    _, _, adj_mx = io_utils.load_adj(args.adj_data)
    adj_mx = np.where(adj_mx > 0.5, 1.0, 0.0)
    
    dataloader = io_utils.load_dataset(args.data, 1)
    
    scaler = dataloader['scaler']
    
    # load model checkpoint
    
    model = models.NBeatsNet(device, args.in_dim, args.out_dim)
    model.to(device)
    
    # load model
    model.load_state_dict(torch.load(args.save + "/G_T_model" + ".pth"))
    
    adj_mx = torch.tensor(adj_mx, dtype=torch.float32).to(device)
    
    # load instance to train
    for i, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        trainx = torch.tensor(x, dtype=torch.float32).to(device)
        trainy = torch.tensor(y, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            _, forecast = model(trainx, adj_mx)
            pred = forecast
            pred_scaled = scaler.inverse_transform(pred)
            
            loss = metrics.masked_mae(pred_scaled, trainy, 0.0)
            print("Original loss:", loss.item())
            
        break
    
    # create the explainer
    explainer = Explainer(device=device,
                          model=model,
                          adj=adj_mx,
                          feat=trainx,
                          pred=pred_scaled,
                          args=args,
                          graph_mode=args.graph_mode,
                          writer=writer,
                          scaler=scaler)
    
    explainer.explain(69)

if __name__ == '__main__':
    main()
    print('-----done-----')