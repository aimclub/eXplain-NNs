import matplotlib
import torch
from typing import Dict, List
from eXNN.InnerNeuralTopology.homologies import InnerNetspaceHomologies, _ComputeBarcode


def ComputeBarcode(data: torch.Tensor,
                   hom_type: str,
                   coefs_type: str) -> matplotlib.figure.Figure:
    """The function plots barcodes.

    Args:
        data (torch.Tensor): input data of shape NxC1x...xCk,
            where N is the number of data points,
            C1,...,Ck are dimensions of each data point
        hom_type (str): homotopy type
        coefs_type (str): coefficients type

    Returns:
        matplotlib.figure.Figure: barcode plot
    """
    return _ComputeBarcode(data, hom_type, coefs_type)


def NetworkHomologies(model: torch.nn.Module,
                      data: torch.Tensor,
                      layers: List[str],
                      hom_type: str,
                      coefs_type: str) -> Dict[str, matplotlib.figure.Figure]:
    """The function plots homology barcodes of latent representations on different levels
        of the neural network.

    Args:
        model (torch.nn.Module): neural network
        data (torch.Tensor): input data of shape NxC1x...xCk,
            where N is the number of data points,
            C1,...,Ck are dimensions of each data point
        layers (List[str]): list of layers for visualization. Defaults to None.
            If None, visualization for all layers is performed
        hom_type (str): homotopy type
        coefs_type (str): coefficients type

    Returns:
        Dict[str, matplotlib.figure.Figure]: dictionary with a barcode plot for each layer
    """
    res = {}
    for layer in layers:
        res[layer] = InnerNetspaceHomologies(model, data, layer, hom_type, coefs_type)
    return res
