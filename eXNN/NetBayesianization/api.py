import torch
import torch.nn as nn
import torch.optim
from NetBayesianization.wrap import create_bayesian_wrapper


def BasicBayesianPrediction(data: torch.Tensor, 
                            model: torch.nn.Module,
                            n_iter: int,
                            mode: str,
                            p: float,
                            a: float,
                            b: float):
    return BasicBayesianWrapper(model, mode, p, a, b).predict(data, n_iter)

class BasicBayesianWrapper:
    def __init__(self,
                 model: torch.nn.Module,
                 mode: str,
                 p: float,
                 a: float,
                 b: float):
        self.model = create_bayesian_wrapper(model, mode, p, a, b)

    
    def predict(self, data, n_iter):
        res = self.model.mean_forward(data, n_iter)
        return {'mean': res[0], 'std': res[1]}
    
    
