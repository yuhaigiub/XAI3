# for nbeats (if you need it)
SEASONALITY_BLOCK = 'seasonality'
TREND_BLOCK = 'trend'
GENERIC_BLOCK = 'generic'

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
        
        self.print_every = 100
        
        self.validate_result = True
        self.test_all = True # if run test.py
        self.epoch_test = 0 # single epoch test
        
        #
        # self.epochs = 5 # epoch to train black box (pre dynamask)
        self.batch_size = 1 # do not change (dynamask only do instance based)
        self.save = 'saved_models/dynamask'
        self.log = 'log/dynamask'
        # -1 if start fresh, else provide the last epoch
        # self.last_epoch = -1
        
        # custom
        self.samples = 10 # number of sample to learn
        self.sample_iters = 2000
        self.initial_mask_coeff = 0.5
        self.keep_ratio = 0.5
        self.size_reg_factor_init = 0.5
        self.size_reg_factor_dilation = 100
        self.time_reg_factor = 0
        
        # custom (black-box)
        self.black_box_file = 'saved_models/nbeats/G_T_model_14.pth'
        
        self.forecast_length=12
        self.backcast_length=12
        
        self.stack_types = (TREND_BLOCK, SEASONALITY_BLOCK, GENERIC_BLOCK)
        self.theta_dims = (4, 8, 1)
        self.number_of_blocks_per_stack = 3
        self.use_graph = False
        