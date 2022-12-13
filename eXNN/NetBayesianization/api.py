from typing import Dict, Optional
import torch
import torch.optim
from eXNN.NetBayesianization.wrap import create_bayesian_wrapper


def BasicBayesianPrediction(data: torch.Tensor,
                            model: torch.nn.Module,
                            n_iter: int,
                            mode: str,
                            p: Optional[float] = None,
                            a: Optional[float] = None,
                            b: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """Function computes mean and standard deviation of bayesian equivalent of a neural network.

    Args:
        data (torch.Tensor): input data of shape NxC1x...xCk, where N is the number of data points, C1,...,Ck are dimensions of each data point
        model (torch.nn.Module): neural network
        n_iter (int): number of samplings form the bayesian equivalent of a neural network
        mode (str): bayesianization method (`basic` or `beta`)
        p (Optional[float], optional): parameter of dropout (for `basic` bayesianization). Defaults to None.
        a (Optional[float], optional): parameter of beta distribution (for `beta` bayesianization). Defaults to None.
        b (Optional[float], optional): parameter of beta distribution (for `beta` bayesianization). Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: dictionary with `mean` and `std` of prediction
    """
    return BasicBayesianWrapper(model, mode, p=p, a=a, b=b).predict(data, n_iter)


class BasicBayesianWrapper:
    def __init__(self,
                 model: torch.nn.Module,
                 mode: str,
                 p: Optional[float] = None,
                 a: Optional[float] = None,
                 b: Optional[float] = None):
        """Class representing bayesian equivalent of a neural network.

        Args:
            model (torch.nn.Module): neural network
            mode (str): bayesianization method (`basic` or `beta`)
            p (Optional[float], optional): parameter of dropout (for `basic` bayesianization). Defaults to None.
            a (Optional[float], optional): parameter of beta distribution (for `beta` bayesianization). Defaults to None.
            b (Optional[float], optional): parameter of beta distribution (for `beta` bayesianization). Defaults to None.
        """
        self.model = create_bayesian_wrapper(model, mode, p=p, a=a, b=b)

    def predict(self, data, n_iter) -> Dict[str, torch.Tensor]:
        """Function computes mean and standard deviation of bayesian equivalent of a neural network.

        Args:
            data (_type_): input data of shape NxC1x...xCk, where N is the number of data points, C1,...,Ck are dimensions of each data point
            n_iter (_type_): number of samplings form the bayesian equivalent of a neural network

        Returns:
            Dict[str, torch.Tensor]: dictionary with `mean` and `std` of prediction
        """
        res = self.model.mean_forward(data, n_iter)
        return {'mean': res[0], 'std': res[1]}
