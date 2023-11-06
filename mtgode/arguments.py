class Arguments():
    def __init__(self):
        self.device = 'cuda' # or cpu
        self.data = 'store/METR-LA'
        self.adj_data = 'store/adj_mx.pkl'
        
        self.in_dims = 2 # number of input channels
        self.out_dims = 2
        self.num_nodes = 207
        
        self.apply_scaling = True
        self.learning_rate = 0.001
        self.dropout = 0.3
        self.weight_decay = 0.0001
        
        self.print_every = 50
        
        self.validate_result = True
        self.test_all = True
        self.epoch_test = 0 # single epoch test
        
        #
        self.epochs = 10
        self.batch_size = 4
        self.save = 'saved_models/mtgode'
        self.log = 'log/mtgode'
        # -1 if start fresh, else provide the last epoch
        self.last_epoch = -1 
        
        # custom
        self.conv_channels = 32
        self.end_channels = 128
        
        self.seq_len = 12
        self.build_A = False
        
        self.time_1 = 1.2 # CTA integration time
        self.time_step_1 = 0.2
        
        self.time_2 = 1.2 # CGP integration time
        self.time_step_2 = 0.2
        
        self.subgraph_size = 20
        self.node_dims = 40
        
        self.alpha = 2.0 # eigen normalization