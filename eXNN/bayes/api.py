from typing import Dict, Optional

import torch
import torch.optim

from eXNN.bayes.wrapper import create_dropout_bayesian_wrapper


class DropoutBayesianWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        mode: str,
        p: Optional[float] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
    ):
        """Class representing bayesian equivalent of a neural network.

        Args:
            model (torch.nn.Module): neural network
            mode (str): bayesianization method (`basic` or `beta`)
            p (Optional[float], optional): parameter of dropout (for `basic`
                bayesianization). Defaults to None.
            a (Optional[float], optional): parameter of beta distribution (for `beta`
                bayesianization). Defaults to None.
            b (Optional[float], optional): parameter of beta distribution (for `beta`
                bayesianization). Defaults to None.
        """
        self.model = create_dropout_bayesian_wrapper(model, mode, p=p, a=a, b=b)

    def predict(self, data, n_iter) -> Dict[str, torch.Tensor]:
        """Function computes mean and standard deviation of bayesian equivalent
            of a neural network.

        Args:
            data (_type_): input data of shape NxC1x...xCk,
                where N is the number of data points,
                C1,...,Ck are dimensions of each data point
            n_iter (_type_): number of samplings form the bayesian equivalent
                of a neural network

        Returns:
            Dict[str, torch.Tensor]: dictionary with `mean` and `std` of prediction
        """
        res = self.model.mean_forward(data, n_iter)
        return {"mean": res[0], "std": res[1]}


class GaussianBayesianWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        sigma: float,
    ):
        """Class representing bayesian equivalent of a neural network.

        Args:
            model (torch.nn.Module): neural network
            sigma (float): std of parameters gaussian noise
        """
        self.model = create_dropout_bayesian_wrapper(model, "gauss", sigma=sigma)

    def predict(self, data, n_iter) -> Dict[str, torch.Tensor]:
        """Function computes mean and standard deviation of bayesian equivalent
            of a neural network.

        Args:
            data (_type_): input data of shape NxC1x...xCk,
                where N is the number of data points,
                C1,...,Ck are dimensions of each data point
            n_iter (_type_): number of samplings form the bayesian equivalent
                of a neural network

        Returns:
            Dict[str, torch.Tensor]: dictionary with `mean` and `std` of prediction
        """
        res = self.model.mean_forward(data, n_iter)
        return {"mean": res[0], "std": res[1]}
