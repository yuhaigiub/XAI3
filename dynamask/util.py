import pickle
import os
from sklearn.metrics import average_precision_score, classification_report

import torch
import numpy as np

class DataLoader(object):
    def __init__(self, xs, ys, batch_size):
        self.batch_size = batch_size
        self.current_ind = -1
        
        num_padding = (batch_size - (len(xs) - batch_size)) % batch_size
        x_padding = np.repeat(xs[-1:], num_padding, axis=0)
        y_padding = np.repeat(ys[-1:], num_padding, axis=0)
        
        xs = np.concatenate([xs, x_padding], axis=0)
        ys = np.concatenate([ys, y_padding], axis=0)
        
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        
        self.xs = xs
        self.ys = ys
    
    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
    
    def get_iterator(self):
        self.current_ind = 0
        
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                
                yield x_i, y_i
                self.current_ind += 1
        
        return _wrapper()

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin')
    except Exception as e:
        print("Unable to load data:", pickle_file)
        print("Error:")
        print(e)
        raise
    
    return pickle_data

def load_adj(pickle_file):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pickle_file)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, apply_scaling=False):
    if valid_batch_size is None:
        valid_batch_size = batch_size
    if test_batch_size is None:
        test_batch_size = batch_size
        
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    
    scaler = None
    if apply_scaling:
        mean = data['x_train'][..., 0].mean()
        std = data['x_train'][..., 0].std()
        scaler = StandardScaler(mean, std)

    
    for category in ['train', 'val', 'test']:
        if apply_scaling:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    
    data['scaler'] = scaler
    
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    # rmse = masked_rmse(pred, real, 0.0).item()
    mse = masked_mse(pred, real, 0.0).item()
    
    return mae, mape, mse