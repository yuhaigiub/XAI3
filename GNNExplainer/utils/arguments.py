class Arguments():
    def __init__(self):
        self.device = 'cuda'
        self.dataset = 'syn1'
        
        self.writer = True
        self.clean_log = True
        self.logdir = 'log/gnn_explainer'
        self.ckptdir = 'saved_models/gnn_explainer/ckpt'
        self.name_suffix = ''
        self.explainer_suffix = ""
        
        self.batch_size = 20
        
        self.input_dims = 10
        self.hidden_dims = 20
        self.output_dims = 20
        
        self.num_gc_layers = 3
        
        self.bias = True
        self.bn = False
        self.dropout = 0.0
        self.clip = 2.0
        
        self.method = 'base'

class ArgumentsTrain(Arguments):
    def __init__(self):
        super(ArgumentsTrain, self).__init__()
        self.epochs = 1000
        self.lr = 0.001
        self.weight_decay = 0.005
        
        self.num_classes = 2
        self.max_nodes = 100
        
        self.train_ratio = 0.8
        self.test_ratio = 0.1
        self.assign_ratio = 0.1

class ArgumentsExplain(Arguments):
    def __init__(self):
        super(ArgumentsExplain, self).__init__()
        self.epochs = 100
        self.lr = 0.1
        self.weight_decay = 0.0

        self.explain_node = 301
        self.graph_idx = -1
        self.graph_mode = False
        self.multigraph_class = -1
        self.multinode_class = -1
        
        self.align_steps = 1000
        self.mask_activation = 'sigmoid'
        self.mask_bias = True
