from model import NBeatsNet

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
        self.test_all = True # if run test.py
        self.epoch_test = 0 # single epoch test
        
        #
        self.epochs = 5
        self.batch_size = 4
        self.save = 'saved_models/nbeats'
        self.log = 'log/nbeats'
        # -1 if start fresh, else provide the last epoch
        self.last_epoch = 9
        
        # custom
        self.forecast_length=12
        self.backcast_length=12
        
        self.stack_types = (NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK, NBeatsNet.GENERIC_BLOCK)
        self.theta_dims = (4, 8, 1)
        self.number_of_blocks_per_stack = 3
        self.use_graph = False
