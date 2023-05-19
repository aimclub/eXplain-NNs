from typing import Dict, List

import matplotlib
import torch

from eXNN.InnerNeuralTopology.homologies import compute_barcode, get_homologies


def ComputeBarcode(
    data: torch.Tensor,
    hom_type: str,
    coefs_type: str,
) -> matplotlib.figure.Figure:
    """This function plots persistent homologies for a cloud of points as barcodes.

    Args:
        data (torch.Tensor): input data of shape NxC1x...xCk,
            where N is the number of data points,
            C1,...,Ck are dimensions of each data point
        hom_type (str): homotopy type
        coefs_type (str): coefficients type

    Returns:
        matplotlib.figure.Figure: barcode plot
    """
    return compute_barcode(data, hom_type, coefs_type)


def NetworkHomologies(
    model: torch.nn.Module,
    data: torch.Tensor,
    layers: List[str],
    hom_type: str,
    coefs_type: str,
) -> Dict[str, matplotlib.figure.Figure]:
    """
    The function plots persistent homologies for latent representations
        on different levels of the neural network as barcodes.

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
        res[layer] = get_homologies(model, data, layer, hom_type, coefs_type)
    return res
