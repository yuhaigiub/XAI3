import torch.optim as optim
from utils.arguments import Arguments

def build_optimizer(args: Arguments, params):
    filter_fn = filter(lambda p : p.requires_grad, params)
    optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    
    # no scheduler (default)
    return None, optimizer