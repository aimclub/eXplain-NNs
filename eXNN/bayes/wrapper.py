import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class ModuleBayesianWrapper(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        p: Optional[float] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
        sigma: Optional[float] = None,
    ):
        super(ModuleBayesianWrapper, self).__init__()

        # Variables correctness checks
        pab_check = "You can either specify the following options (exclusively):\n\
              - p (simple dropout)\n - a and b (beta dropout)\n - sigma (gaussian dropout)"
        assert (p is not None and a is None and b is None and sigma is None) or \
               (p is None and a is not None and b is not None and sigma is None) or \
               (p is None and a is None and b is None and sigma is not None), pab_check

        if (p is None) and (sigma is None):
            ab_check = "If you choose to specify a and b, you must specify both"
            assert (a is not None) and (b is not None), ab_check

        # At the moment we are only modifying linear and convolutional layers, so check this
        self.layer = layer

        if type(layer) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            self.p, self.a, self.b, self.sigma = p, a, b, sigma

    def augment_weights(self, weights, bias):

        # Check if dropout is chosen
        if (self.p is not None) or (self.a is not None and self.b is not None):
            # Select correct option and apply dropout
            if self.p is not None:
                p = self.p
            else:
                p = Beta(torch.tensor(self.a), torch.tensor(self.b)).sample()

            weights = F.dropout(weights, p, training=True)
            bias = F.dropout(bias, p, training=True)

        else:
            # If gauss is chosen, then apply it
            weights = weights + (torch.randn(*weights.shape) * self.sigma).to(weights.device)
            bias = bias + (torch.randn(*bias.shape) * self.sigma).to(bias.device)

        return weights, bias

    def forward(self, x):

        weight, bias = self.augment_weights(self.layer.weight, self.layer.bias)

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
        self.model = replace_modules_with_wrapper(
            self.model,
            ModuleBayesianWrapper,
            {"p": dropout_p},
        )

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
        self.model = replace_modules_with_wrapper(
            self.model,
            ModuleBayesianWrapper,
            {"a": alpha, "b": beta},
        )

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


class NetworkBayesGauss(nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        sigma: float,
    ):

        super(NetworkBayesGauss, self).__init__()
        self.model = copy.deepcopy(model)
        self.model = replace_modules_with_wrapper(
            self.model,
            ModuleBayesianWrapper,
            {"sigma": sigma},
        )

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
    sigma: Optional[float] = None,
) -> torch.nn.Module:
    if mode == "basic":
        net = NetworkBayes(model, p)

    elif mode == "beta":
        net = NetworkBayesBeta(model, a, b)

    elif mode == 'gauss':
        net = NetworkBayesGauss(model, sigma)

    return net
