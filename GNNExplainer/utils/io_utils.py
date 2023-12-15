""" 
io_utils.py
Utilities for reading and writing logs.
"""
import os
import torch
from utils.arguments import Arguments

def load_ckpt(args: Arguments, is_best=False):
    '''
    Load a pre-trained pytorch model from checkpoint
    '''
    print('Loading model')
    
    filename = create_filename(args.ckptdir, args, is_best)
    print('filename:', filename)
    
    if os.path.isfile(filename):
        print('=> Loading checkpoint "{}"'.format(filename))
        ckpt = torch.load(filename)
    else:
        raise Exception("File Not Found")
    
    return ckpt

def save_ckpt(model, 
              optimizer, 
              args: Arguments, 
              num_epochs=-1, 
              isbest=False, 
              cg_dict=None):
    """
    Save pytorch model checkpoint.

    Args:
        - model         : The PyTorch model to save.
        - optimizer     : The optimizer used to train the model.
        - args          : A dict of meta-data about the model.
        - num_epochs    : Number of training epochs.
        - isbest        : True if the model has the highest accuracy so far.
        - cg_dict       : A dictionary of the sampled computation graphs.
    """
    
    filename = create_filename(args.ckptdir, args, isbest, num_epochs=num_epochs)
    torch.save({
        'epochs': num_epochs,
        'model_type': args.method,
        'optimizer': optimizer,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'cg': cg_dict,
    }, filename)

def create_filename(save_dir, args: Arguments, is_best=False, num_epochs=-1):
    filename = os.path.join(save_dir, gen_prefix(args))
    # os.makedirs(filename, exist_ok=True)
    
    if is_best:
        filename = os.path.join(filename, 'best')
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))
    
    return filename + '.pth.tar'

def gen_prefix(args: Arguments):
    '''
    Generate label prefix for a graph model.
    '''
    name = args.dataset
    
    name += "_" + args.method
    name += "_h" + str(args.hidden_dims) + "_o" + str(args.output_dims)
    
    if not args.bias:
        name += "_nobias"
    elif len(args.name_suffix) > 0:
        name += "_" + args.name_suffix
    
    return name

def gen_explainer_prefix(args: Arguments):
    '''
    Generate label prefix for a graph explainer model.
    '''
    name = gen_prefix(args) + "_explain"
    
    if len(args.explainer_suffix) > 0:
        name += '_' + args.explainer_suffix
    
    return name