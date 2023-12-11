""" 
explainer_main.py
Main user interface for the explainer module.
"""
import warnings
warnings.filterwarnings('ignore')

import torch

import os
import sys
import shutil

from tensorboardX import SummaryWriter

import utils.io_utils as io_utils
import models
import explain
from utils.arguments import ArgumentsExplain

def main():
    args = ArgumentsExplain()
    device = torch.device(args.device)
    
    # configure the logging directory
    # if args.writer:
    #     path = os.path.join(args.logdir, io_utils.gen_explainer_prefix(args))
    #     if os.path.isdir(path) and args.clean_log:
    #         print('Removing existing log dir:', path)
    #         shutil.rmtree(path)
    #     writer = SummaryWriter(path)
    # else:
    #     writer = None
    
    # load model checkpoint
    ckpt = io_utils.load_ckpt(args)
    cg_dict = ckpt['cg'] # get computation graph
    in_dims = cg_dict['feat'].shape[2]
    num_classes = cg_dict['pred'].shape[2]
    
    print("Loaded model from {}".format(args.ckptdir))
    print("input dim: ", in_dims, "; num classes: ", num_classes)
    
    # determine explainer mode
    graph_mode = (args.graph_mode
                  or args.multigraph_class >= 0
                  or args.graph_idx >= 0)
    print('graph mode:', graph_mode)
    
    # build model
    print("Method: ", args.method)
    model = models.GCNEncoderNode(device,
                                  in_dims=in_dims, 
                                  hidden_dims=args.hidden_dims, 
                                  embedding_dims=args.output_dims, 
                                  label_dims=num_classes,
                                  num_layers=args.num_gc_layers,
                                  bn=args.bn,
                                  dropout=args.dropout,
                                  bias=args.bias)
    # load state_dict
    model.load_state_dict(ckpt['model_state'])
    
    # create explainer
    explainer = explain.Explainer(
        device=device,
        model=model,
        adj=cg_dict['adj'],
        feat=cg_dict['feat'],
        labels=cg_dict['labels'],
        pred=cg_dict['pred'],
        train_idx=cg_dict['train_idx'],
        graph_idx=args.graph_idx,
        args=args,
        writer=None)
    
    if args.explain_node is not None:
        explainer.explain(args.explain_node, unconstrained=False)
    elif args.graph_mode:
        raise Exception("mode not available")
    else:
        raise Exception("mode not available")

if __name__ == '__main__':
    main()
    print('-----done-----')