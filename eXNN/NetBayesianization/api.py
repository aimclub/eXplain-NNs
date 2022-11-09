import torch
import torch.nn as nn
import copy
import statistics
import torch.optim
from torch.distributions import Beta

#calculate mean and std after applying bayesian
class NetworkBayesWrapper(nn.Module):
    def __init__(self, 
                 model: nn.Module, 
                 dropout_p: float):
        
        super(NetworkBayesWrapper, self).__init__()
        self.model = model
        self.dropout_p = dropout_p
         
    def mean_forward(self, 
                     data: torch.Tensor, 
                     n_iter: int):
        
        results = []
        out = {}
        for i in range(n_iter):
            model_copy = copy.deepcopy(self.model)
            state_dict = model_copy.state_dict()
            state_dict_v2 = copy.deepcopy(state_dict)
            for key, value in state_dict_v2.items():
                if 'weight' in key:
                    output = nn.functional.dropout(value, self.dropout_p, training=True)
                    state_dict_v2[key] = output
            model_copy.load_state_dict(state_dict_v2, strict=True)
            output = model_copy(data)
            results.append(output)
        result = [tensor.item() for tensor in results]
        out['mean'] = round(statistics.mean(result), 3)
        out['std'] = round(statistics.stdev(result), 3)
        return out
        
    

#calculate mean and std after applying bayesian with beta distribution
class NetworkBayesWrapperBeta(nn.Module):
    def __init__(self, 
                 model: torch.nn.Module,
                 alpha: float, 
                 beta: float):
        
        super(NetworkBayesWrapperBeta, self).__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
         
    def mean_forward(self, 
                     x: torch.Tensor, 
                     n_iter: int):
        
        results = []
        out = {}
        m = Beta(torch.tensor(self.alpha), torch.tensor(self.beta))
        for i in range(n_iter):
            p = m.sample()
            model_copy = copy.deepcopy(self.model)
            state_dict = model_copy.state_dict()
            state_dict_v2 = copy.deepcopy(state_dict)
            for key, value in state_dict_v2.items():
                if 'weight' in key:
                    output = nn.functional.dropout(value, p, training=True)
                    state_dict_v2[key] = output
            model_copy.load_state_dict(state_dict_v2, strict=True)
            output = model_copy(x)
            results.append(output)
        result = [tensor.item() for tensor in results]
        out['mean'] = round(statistics.mean(result), 3)
        out['std'] = round(statistics.stdev(result), 3)
        return out  
