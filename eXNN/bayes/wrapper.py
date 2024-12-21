import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import Beta


class ModuleBayesianWrapper(nn.Module):
    """
    A wrapper for neural network layers to apply Bayesian-style dropout or noise during training.

    Args:
        layer (nn.Module): The layer to wrap (e.g., nn.Linear, nn.Conv2d).
        p (Optional[float]): Dropout probability for simple dropout. Mutually exclusive with `a`, `b`, and `sigma`.
        a (Optional[float]): Alpha parameter for Beta distribution dropout. Used with `b`.
        b (Optional[float]): Beta parameter for Beta distribution dropout. Used with `a`.
        sigma (Optional[float]): Standard deviation for Gaussian noise. Mutually exclusive with `p`, `a`, and `b`.
    """

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
        """
        Apply the specified noise or dropout to the weights and bias.

        Args:
            weights (torch.Tensor): The weights of the layer.
            bias (torch.Tensor): The bias of the layer (can be None).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The augmented weights and bias.
        """

        # Check if dropout is chosen
        if (self.p is not None) or (self.a is not None and self.b is not None):
            # Select correct option and apply dropout
            if self.p is not None:
                p = self.p
            else:
                p = Beta(torch.tensor(self.a), torch.tensor(self.b)).sample()

            weights = functional.dropout(weights, p, training=True)
            if bias is not None:
                # In layers we sometimes have the ability to set bias to None
                bias = functional.dropout(bias, p, training=True)

        else:
            # If gauss is chosen, then apply it
            weights = weights + (torch.randn(*weights.shape) * self.sigma).to(weights.device)
            if bias is not None:
                # In layers we sometimes have the ability to set bias to None
                bias = bias + (torch.randn(*bias.shape) * self.sigma).to(bias.device)

        return weights, bias

    def forward(self, x):
        """
        Forward pass through the layer with augmented weights.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        weight, bias = self.augment_weights(self.layer.weight, self.layer.bias)

        if isinstance(self.layer, nn.Linear):
            return functional.linear(x, weight, bias)
        elif type(self.layer) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            return self.layer._conv_forward(x, weight, bias)
        else:
            return self.layer(x)


def replace_modules_with_wrapper(model, wrapper_module, params):
    """
    Recursively replaces layers in a model with a Bayesian wrapper.

    Args:
        model (nn.Module): The model containing layers to replace.
        wrapper_module (type): The wrapper class.
        params (dict): Parameters for the wrapper.

    Returns:
        nn.Module: The model with wrapped layers.
    """

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
    """
    Bayesian network with standard dropout.

    Args:
        model (nn.Module): The base model.
        dropout_p (float): Dropout probability.
    """

    def __init__(
            self,
            model: nn.Module,
            dropout_p: float,
    ):
        """
        Initialize the NetworkBayes with standard dropout.

        Args:
            model (nn.Module): The base model to wrap with Bayesian dropout.
            dropout_p (float): Dropout probability for the Bayesian wrapper.
        """
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
        """
        Perform forward passes to estimate the mean and standard deviation of outputs.

        Args:
            data (torch.Tensor): Input tensor.
            n_iter (int): Number of stochastic forward passes.

        Returns:
            torch.Tensor: A tensor containing the mean (dim=0) and standard deviation (dim=1) of outputs.
        """
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
    """
    Bayesian network with Beta distribution dropout.

    Args:
        model (nn.Module): The base model.
        alpha (float): Alpha parameter for the Beta distribution.
        beta (float): Beta parameter for the Beta distribution.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            alpha: float,
            beta: float,
    ):
        """
        Initialize the NetworkBayesBeta with Beta distribution dropout.

        Args:
            model (nn.Module): The base model to wrap with Bayesian Beta dropout.
            alpha (float): Alpha parameter of the Beta distribution.
            beta (float): Beta parameter of the Beta distribution.
        """
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
        """
        Perform forward passes to estimate the mean and standard deviation of outputs.

        Args:
            data (torch.Tensor): Input tensor.
            n_iter (int): Number of stochastic forward passes.

        Returns:
            torch.Tensor: A tensor containing the mean (dim=0) and standard deviation (dim=1) of outputs.
        """
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
    """
    Bayesian network with Gaussian noise.

    Args:
        model (nn.Module): The base model.
        sigma (float): Standard deviation of the Gaussian noise.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            sigma: float,
    ):
        """
        Initialize the NetworkBayesGauss with Gaussian noise.

        Args:
            model (nn.Module): The base model to wrap with Bayesian Gaussian noise.
            sigma (float): Standard deviation of the Gaussian noise to apply.
        """
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
        """
        Perform forward passes to estimate the mean and standard deviation of outputs.

        Args:
            data (torch.Tensor): Input tensor.
            n_iter (int): Number of stochastic forward passes.

        Returns:
            torch.Tensor: A tensor containing the mean (dim=0) and standard deviation (dim=1) of outputs.
        """
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
    """
    Creates a Bayesian network with the specified dropout mode.

    Args:
        model (nn.Module): The base model.
        mode (str): The dropout mode ("basic", "beta", "gauss").
        p (Optional[float]): Dropout probability for "basic" mode.
        a (Optional[float]): Alpha parameter for "beta" mode.
        b (Optional[float]): Beta parameter for "beta" mode.
        sigma (Optional[float]): Standard deviation for "gauss" mode.

    Returns:
        nn.Module: The Bayesian network.
    """

    if mode == "basic":
        net = NetworkBayes(model, p)

    elif mode == "beta":
        net = NetworkBayesBeta(model, a, b)

    elif mode == 'gauss':
        net = NetworkBayesGauss(model, sigma)

    else:
        raise ValueError("Mode should be one of ('basic', 'beta', 'gauss').")

    return net
