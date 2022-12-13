from typing import Optional
import torch
import torch.nn as nn
import copy
import torch.optim
from torch.distributions import Beta

# calculate mean and std after applying bayesian


class NetworkBayes(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 dropout_p: float):

        super(NetworkBayes, self).__init__()
        self.model = model
        self.dropout_p = dropout_p

    def mean_forward(self,
                     data: torch.Tensor,
                     n_iter: int):

        results = []
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

        results = torch.stack(results, dim=1)
        results = torch.stack([
            torch.mean(results, dim=1),
            torch.std(results, dim=1)
        ], dim=0)
        return results


# calculate mean and std after applying bayesian with beta distribution
class NetworkBayesBeta(nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 alpha: float,
                 beta: float):

        super(NetworkBayesBeta, self).__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta

    def mean_forward(self,
                     data: torch.Tensor,
                     n_iter: int):

        results = []
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
            output = model_copy(data)
            results.append(output)
        results = torch.stack(results, dim=1)
        results = torch.stack([
            torch.mean(results, dim=1),
            torch.std(results, dim=1)
        ], dim=0)
        return results


def create_bayesian_wrapper(model: torch.nn.Module,
                            mode: Optional[str] = 'basic',
                            p: Optional[float] = None,
                            a: Optional[float] = None,
                            b: Optional[float] = None) -> torch.nn.Module:
    if mode == 'basic':
        net = NetworkBayes(model, p)

    elif mode == 'beta':
        net = NetworkBayesBeta(model, a, b)

    return net
