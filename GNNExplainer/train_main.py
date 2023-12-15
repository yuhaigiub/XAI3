"""
train.py
Main interface to train the GNNs that will be later explained.
"""

import warnings
warnings.filterwarnings('ignore')

import time
import os

import numpy as np
import sklearn.metrics as metrics

import torch
from torch import nn

from typing import List
from networkx.classes.graph import Graph

from tensorboardX import SummaryWriter

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import gengraph
import models
import utils.feature_generator as feat_gen
from utils.arguments import ArgumentsTrain

def main():
    args = ArgumentsTrain()
    device = torch.device(args.device)
    
    path = os.path.join(args.logdir, io_utils.gen_explainer_prefix(args))
    writer = SummaryWriter(path)
    
    if args.dataset == 'syn1':
        syn_task1(args, device, writer=writer)
    
    writer.close()

def syn_task1(args: ArgumentsTrain, device, writer):
    # data
    feature_generator = feat_gen.ConstFeatureGen(np.ones(args.input_dims, dtype=np.float32))
    G, labels, name = gengraph.gen_syn1(feature_generator=feature_generator)
    num_classes = max(labels) + 1
    
    print(G)
    print('number of classes:', num_classes)
    print('labels size:', len(labels))
    
    if args.method == 'att':
        raise Exception('Attribution method not implemented')
    else:
        print('Method:', args.method)
        model = models.GCNEncoderNode(device,
                                      in_dims=args.input_dims, 
                                      hidden_dims=args.hidden_dims, 
                                      embedding_dims=args.output_dims, 
                                      label_dims=num_classes,
                                      num_layers=args.num_gc_layers,
                                      bn=args.bn,
                                      dropout=args.dropout,
                                      bias=args.bias).to(device)
    
    train_node_classifier(device, G, labels, model, args, writer=writer)

def train_node_classifier(device,
                          G: Graph, 
                          labels: List[int], 
                          model: models.GCNEncoderGraph | models.GCNEncoderNode, 
                          args: ArgumentsTrain, 
                          writer=None):
    # train/test split only for nodes
    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]
    
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]
    
    data = gengraph.preprocess_input_graph(G, labels)
    labels_train = torch.tensor(data['labels'][:, train_idx], dtype=torch.long).to(device)
    adj = torch.tensor(data['adj'], dtype=torch.float32).to(device)
    x = torch.tensor(data['feat'], dtype=torch.float32, requires_grad=True).to(device)
    
    scheduler, optimizer = train_utils.build_optimizer(args, model.parameters())
    print('number of parameters:', len(list(model.parameters())))
    
    model.train()
    ypred = None
    for epoch in range(args.epochs):
        begin_time = time.time()
        model.zero_grad()
        
        ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]

        loss = model.loss(ypred_train, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        
        elapsed = time.time() - begin_time
        
        # evaluate
        result_train, result_test = evaluate_node(ypred.cpu(), data["labels"], train_idx, test_idx)
        
        # write logs
        if writer is not None:
            pass
        
        # schedule
        if scheduler is not None:
            scheduler.step()
        
        
        if epoch % 10 == 0:
            print(
                "epoch:",      epoch,                          "; ",
                "loss:",       round(loss.item(), 3),          "; ",
                "train_acc:",  round(result_train["acc"], 3),  "; ",
                "test_acc:",   round(result_test["acc"], 3),   "; ",
                "train_prec:", round(result_train["prec"], 3), "; ",
                "test_prec:",  round(result_test["prec"], 3),  "; ",
                "epoch time:", "{0:0.2f}".format(elapsed),
            )
        
    print(result_train["conf_mat"])
    print(result_test["conf_mat"])
    
    # computation graph
    model.eval()
    ypred, _ = model(x, adj)
    cg_data = {
        'adj': data['adj'],
        'feat': data['feat'],
        'labels': data['labels'],
        'pred': ypred.cpu().detach().numpy(),
        'train_idx': train_idx,
    }
    io_utils.save_ckpt(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)

def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()
    
    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])
    
    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test

if __name__ == '__main__':
    main()
    print('-----done-----')
    