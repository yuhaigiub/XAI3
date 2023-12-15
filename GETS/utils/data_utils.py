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
