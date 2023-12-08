from typing import Dict, List, Optional, Union

import matplotlib
import numpy as np
import torch

from eXNN.topology import homologies, metrics


def get_data_barcode(
    data: torch.Tensor,
    hom_type: str,
    coefs_type: str,
) -> Dict[str, np.ndarray]:
    """This function computes persistent homologies for a cloud of points as barcodes.

    Args:
        data (torch.Tensor): input data of shape NxC1x...xCk,
            where N is the number of data points,
            C1,...,Ck are dimensions of each data point
        hom_type (str): homotopy type
        coefs_type (str): coefficients type

    Returns:
        Dict[str, np.ndarray]: barcode
    """
    return homologies.compute_data_barcode(data, hom_type, coefs_type)


def get_nn_barcodes(
    model: torch.nn.Module,
    data: torch.Tensor,
    layers: List[str],
    hom_type: str,
    coefs_type: str,
) -> Dict[str, Dict[str, np.ndarray]]:
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
        Dict[str, Dict[str, np.ndarray]]: dictionary with a barcode for each layer
    """
    res = {}
    for layer in layers:
        res[layer] = homologies.compute_nn_barcode(model, data, layer, hom_type, coefs_type)
    return res


def plot_barcode(barcode: Dict[str, np.ndarray]) -> matplotlib.figure.Figure:
    """
    The function creates a plot of a persistent homologies barcode.

    Args:
        barcode (Dict[str, np.ndarray]): barcode

    Returns:
        matplotlib.figure.Figure: a plot of the barcode
    """
    return homologies.plot_barcode(barcode)


def evaluate_barcode(
    barcode: Dict[str, np.ndarray], metric_name: Optional[str] = None
) -> Union[float, Dict[str, float]]:
    """
    The function evaluates a persistent homologies barcode with a metric.

    Args:
        barcode (Dict[str, np.ndarray]): barcode
        metric_name (Optional[str]): metric name
            (if `None` all available metrics values are computed)

    Returns:
        Union(float, Dict[str, float]): float if metric is specified
            or a dictionary with value of each available metric
    """
    return metrics.compute_metric(barcode, metric_name)
