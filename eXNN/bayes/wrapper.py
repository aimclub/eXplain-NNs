import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from typing import Optional

class DropoutBayesianWrapper(nn.Module):
    def __init__(
        self, 
        layer: nn.Module, 
        p: Optional[float] = None, 
        a: Optional[float] = None, 
        b: Optional[float] = None
    ):
        super(DropoutBayesianWrapper, self).__init__()

        assert (p is not None) != ((a is not None) and (b is not None)), "You can either specify p (simple dropout), or specify a and b (beta dropout)"
        
        if not type(layer) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            # At the moment we are only modifying linear and convolutional layers
            self.layer = layer
        else:
            self.layer = layer

            self.p = p
            self.a, self.b = a, b

            if self.p is None:
                assert (self.a is not None) and (self.b is not None), "If you choose to specify a and b, you must to specify both"

    def dropout_weights(self, weights, bias):
        if self.p is not None:
            p = self.p
        else:
            p = Beta(torch.tensor(self.a), torch.tensor(self.b)).sample()

        weights = F.dropout(weights, p, training=True)
        bias = F.dropout(bias, p, training=True)

        return weights, bias

    def forward(self, x):

        weight, bias = self.dropout_weights(self.layer.weight, self.layer.bias)

        if isinstance(self.layer, nn.Linear):
            return F.linear(x, weight, bias)
        elif type(self.layer) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            return self.layer._conv_forward(x, weight, bias)
        else:
            return self.layer(x)


def replace_modules_with_wrapper(model, wrapper_module, params):
    if type(model) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        return wrapper_module(model, **params)

    elif isinstance(model, nn.Sequential):
        modules = []
        for module in model:
            modules.append(replace_modules_with_wrapper(module, wrapper_module, params))
        return nn.Sequential(*modules)

    elif isinstance(model, nn.Module):
        for name, module in model.named_children():
            setattr(model, name, replace_modules_with_wrapper(module, wrapper_module, params))
        return model


class NetworkBayes(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dropout_p: float,
    ):

        super(NetworkBayes, self).__init__()
        self.model = copy.deepcopy(model)
        self.model = replace_modules_with_wrapper(self.model, DropoutBayesianWrapper, {"p": dropout_p})

    def mean_forward(
        self,
        data: torch.Tensor,
        n_iter: int,
    ):

        results = []
        for _ in range(n_iter):
            results.append(self.model.forward(data))
        

        results = torch.stack(results, dim=1)

        results = torch.stack(
            [
                torch.mean(results, dim=1),
                torch.std(results, dim=1),
            ],
            dim=0,
        )
        return results


# calculate mean and std after applying bayesian with beta distribution
class NetworkBayesBeta(nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        alpha: float,
        beta: float,
    ):

        super(NetworkBayesBeta, self).__init__()
        self.model = copy.deepcopy(model)
        self.model = replace_modules_with_wrapper(self.model, DropoutBayesianWrapper, {"a": alpha, "b": beta})

    def mean_forward(
        self,
        data: torch.Tensor,
        n_iter: int,
    ):

        results = []
        for _ in range(n_iter):
            results.append(self.model.forward(data))
        

        results = torch.stack(results, dim=1)

        results = torch.stack(
            [
                torch.mean(results, dim=1),
                torch.std(results, dim=1),
            ],
            dim=0,
        )
        return results


def create_dropout_bayesian_wrapper(
    model: torch.nn.Module,
    mode: Optional[str] = "basic",
    p: Optional[float] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
) -> torch.nn.Module:
    if mode == "basic":
        net = NetworkBayes(model, p)

    elif mode == "beta":
        net = NetworkBayesBeta(model, a, b)

    return net

