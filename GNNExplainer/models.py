import torch
from torch.nn import init
from torch import nn, Tensor
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self,
                 device,
                 in_dims,
                 out_dims,
                 dropout=0.0,
                 bias=True,
                 normalize_embedding=True):
        super(GraphConv, self).__init__()
        self.device = device
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        
        self.normalize_embedding = normalize_embedding
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.weight = nn.Parameter(torch.FloatTensor(in_dims, out_dims))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dims))
        else:
            self.bias = None
    
    def forward(self, x: Tensor, adj: Tensor):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        
        y = torch.matmul(adj, x).to(self.device)
        y = torch.matmul(y, self.weight)
        
        if self.bias is not None:
            y = y + self.bias
        
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        
        return y, adj

class GCNEncoderGraph(nn.Module):
    def __init__(self,
                 device,
                 in_dims,
                 hidden_dims,
                 embedding_dims,
                 label_dims,
                 num_layers,
                 pred_hidden_dims_list=[],
                 concat=True,
                 dropout=0.0,
                 bn=True, 
                 bias=True):
        super(GCNEncoderGraph, self).__init__()
        self.device = device
        self.bn = bn
        
        self.num_layers = num_layers
        self.num_aggs = 1
        self.concat = concat
        
        self.bias = bias
        
        self.conv_first = GraphConv(device, in_dims, hidden_dims, dropout, bias)
        
        self.conv_block = nn.ModuleList()
        for i in range(num_layers - 2):
            self.conv_block.append(GraphConv(device, hidden_dims, hidden_dims, dropout, bias))
        
        self.conv_last = GraphConv(device, hidden_dims, embedding_dims, dropout, bias)
        
        self.activation = nn.ReLU()
        
        self.label_dims = label_dims
        self.pred_input_dims = embedding_dims
        
        if concat:
            self.pred_input_dims = hidden_dims * (num_layers - 1) + embedding_dims
        else:
            self.pred_input_dims = embedding_dims

        pred_input_dims = self.pred_input_dims * self.num_aggs
        if(len(pred_hidden_dims_list) == 0):
            self.pred_model = nn.Linear(pred_input_dims, label_dims)
        else:
            pred_layers = []
            for pred_dims in pred_hidden_dims_list:
                pred_layers.append(nn.Linear(pred_input_dims, pred_dims))
                pred_layers.append(self.activation)
                pred_input_dims = pred_dims
            pred_layers.append(nn.Linear(pred_input_dims, label_dims))
            self.pred_model = nn.Sequential(*pred_layers)
        
        # initialize parameters
        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=init.calculate_gain('relu'))
                
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        
        self.to(device)
    
    def apply_bn(self, x: Tensor):
        bn_module = nn.BatchNorm1d(x.size()[1])
        return bn_module(x)
    
    def construct_mask(self, max_nodes, batch_num_nodes):
        pass
    
    def gcn_forward(self, x, adj, embedding_mask):
        x, adj_att = self.conv_first(x, adj)
        x = self.activation(x)
        if self.bn:
            x = self.apply_bn(x)
        
        x_all = [x]
        adj_att_all = [adj_att]
        
        for i in range(len(self.conv_block)):
            x, _ = self.conv_block[i](x, adj)
            x = self.activation(x)
            if self.bn:
                x = self.apply_bn(x)
            
            x_all.append(x)
            adj_att_all.append(adj_att)
        
        x, adj_att = self.conv_last(x, adj)
        
        x_all.append(x)
        adj_att_all.append(adj_att)
        
        if self.concat:
            x_tensor = torch.cat(x_all, dim=2)
        else:
            x_tensor = x
        
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        self.embedding_tensor = x_tensor
        
        # adj_att_tensor: [batch_size, num_nodes, hidden_dims * num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)
        
        return x_tensor, adj_att_tensor
    
    def forward(self, x, adj, batch_num_nodes=None):
        raise Exception("You haven't implement this yet")
    
    def loss(self, pred, label, type="softmax"):
        raise Exception("You haven't implement this yet")

class GCNEncoderNode(GCNEncoderGraph):
    def __init__(self,
                 device,
                 in_dims,
                 hidden_dims,
                 embedding_dims,
                 label_dims,
                 num_layers,
                 pred_hidden_dims=[],
                 concat=True,
                 dropout=0.0,
                 bn=True,
                 bias=True):
        super(GCNEncoderNode, self).__init__(device,
                                             in_dims,
                                             hidden_dims,
                                             embedding_dims,
                                             label_dims,
                                             num_layers,
                                             pred_hidden_dims,
                                             concat,
                                             dropout, 
                                             bn, 
                                             bias)
        self.celoss = nn.CrossEntropyLoss()
        self.to(device)
    
    def forward(self, x, adj, batch_num_nodes=None):
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes).to(self.device)
        else:
            embedding_mask = None
        
        self.embedding_tensor, adj_att = self.gcn_forward(x, adj, embedding_mask)
        pred = self.pred_model(self.embedding_tensor)
        
        return pred, adj_att
    
    def loss(self, pred, label):
        pred = torch.transpose(pred, 1, 2)
        return self.celoss(pred, label)