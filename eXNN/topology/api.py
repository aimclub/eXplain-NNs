from typing import Dict, List, Optional, Union

import matplotlib
import numpy as np
import torch

from eXNN.topology import homologies, metrics


def get_data_barcode(
    data: torch.Tensor,
    hom_type: str,
    coefficient_type: str,
) -> Dict[str, np.ndarray]:
    """
    Computes persistent homologies for a cloud of points as barcodes.

    Args:
        data (torch.Tensor): Input data of shape NxC1x...xCk,
            where N is the number of data points,
            C1,...,Ck are dimensions of each data point.
        hom_type (str): Homotopy type.
        coefficient_type (str): Coefficients type.

    Returns:
        Dict[str, np.ndarray]: Barcode.
    """
    return homologies.compute_data_barcode(data, hom_type, coefficient_type)


def get_nn_barcodes(
    model: torch.nn.Module,
    data: torch.Tensor,
    layers: List[str],
    hom_type: str,
    coefficient_type: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Computes persistent homologies for latent representations on different
    levels of the neural network as barcodes.

    Args:
        model (torch.nn.Module): Neural network.
        data (torch.Tensor): Input data of shape NxC1x...xCk,
            where N is the number of data points,
            C1,...,Ck are dimensions of each data point.
        layers (List[str]): List of layers for visualization.
        hom_type (str): Homotopy type.
        coefficient_type (str): Coefficients type.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: Dictionary with a barcode for each layer.
    """
    res = {}
    for layer in layers:
        res[layer] = homologies.compute_nn_barcode(model, data, layer, hom_type, coefficient_type)
    return res


def plot_barcode(barcode: Dict[str, np.ndarray]) -> matplotlib.figure.Figure:
    """
    Creates a plot of a persistent homologies barcode.

    Args:
        barcode (Dict[str, np.ndarray]): Barcode.

    Returns:
        matplotlib.figure.Figure: Plot of the barcode.
    """
    return homologies.plot_barcode(barcode)


def evaluate_barcode(
    barcode: Dict[str, np.ndarray], metric_name: Optional[str] = None
) -> Union[float, Dict[str, float]]:
    """
    Evaluates a persistent homologies barcode with a metric.

    Args:
        barcode (Dict[str, np.ndarray]): Barcode.
        metric_name (Optional[str]): Metric name (if None, all available metrics
            values are computed).

    Returns:
        Union[float, Dict[str, float]]: Float if metric is specified, or a
        dictionary with values of each available metric.
    """
    return metrics.compute_metric(barcode, metric_name)
