from typing import Optional
import torch
import torch.optim
from eXNN.NetBayesianization.wrap import create_bayesian_wrapper


def BasicBayesianPrediction(data: torch.Tensor, 
                            model: torch.nn.Module,
                            n_iter: int,
                            mode: str,
                            p: Optional[float]=None,
                            a: Optional[float]=None,
                            b: Optional[float]=None):
    return BasicBayesianWrapper(model, mode, p=p, a=a, b=b).predict(data, n_iter)

class BasicBayesianWrapper:
    def __init__(self,
                 model: torch.nn.Module,
                 mode: str,
                 p: Optional[float]=None,
                 a: Optional[float]=None,
                 b: Optional[float]=None):
        self.model = create_bayesian_wrapper(model, mode, p=p, a=a, b=b)

    
    def predict(self, data, n_iter):
        res = self.model.mean_forward(data, n_iter)
        return {'mean': res[0], 'std': res[1]}
    
    
