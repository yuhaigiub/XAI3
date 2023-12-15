class Arguments:
    def __init__(self):
        self.writer = True
        self.save = 'saved_models/GETS'
        self.device = 'cuda'
        self.data = 'store/METR-LA'
        self.adj_data = 'store/adj_mx.pkl'
        
        self.clean_log = False
        self.logdir = 'log/GETS'
        
        self.num_nodes = 207
        self.in_dim = 2
        self.out_dim = 2
        
        self.num_gc_layers = 9
        
        self.bias = True

class ArgumentsTrain(Arguments):
    def __init__(self):
        super(ArgumentsTrain, self).__init__()
        self.epochs = 5
        self.batch_size = 16
        
        self.lr = 0.001
        self.weight_decay = 0.05


class ArgumentsExplain(Arguments):
    def __init__(self):
        super(ArgumentsExplain, self).__init__()
        self.epochs = 300
        self.batch_size = 1
        
        self.lr = 0.1
        self.weight_decay = 0.0
        
        self.graph_mode = False
        self.mask_bias = True
        self.mask_activation = 'sigmoid'