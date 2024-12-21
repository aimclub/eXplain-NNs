from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from gtda.homology import (
    SparseRipsPersistence,
    VietorisRipsPersistence,
    WeakAlphaPersistence,
)


def _get_activation(model: torch.nn.Module, x: torch.Tensor, layer: str) -> torch.Tensor:
    """
    Extracts the activation output of a specified layer from a given model.

    Args:
        model (torch.nn.Module): The neural network model.
        x (torch.Tensor): The input tensor to the model.
        layer (str): The name of the layer to extract activation from.

    Returns:
        torch.Tensor: The activation output of the specified layer.
    """
    activation = {}

    def store_output(name):
        def hook(_, __, output):
            activation[name] = output.detach()

        return hook

    hook_handle = getattr(model, layer).register_forward_hook(store_output(layer))
    model.forward(x)
    hook_handle.remove()
    return activation[layer]


def _diagram_to_barcode(plot) -> Dict[str, np.ndarray]:
    """
    Converts a persistence diagram plot into a barcode representation.

    Args:
        plot: The plot object containing persistence diagram data.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are homology types, and values are arrays of intervals.
    """
    data = plot["data"]
    homologies = {}
    for homology in data:
        if homology["name"] is None:
            continue
        homologies[homology["name"]] = list(zip(homology["x"], homology["y"]))

    for key in homologies.keys():
        homologies[key] = sorted(homologies[key], key=lambda x: x[0])
    return homologies


def plot_barcode(barcode: Dict[str, np.ndarray]) -> plt.Figure:
    """
    Plots a barcode diagram from given intervals.

    Args:
        barcode (Dict[str, np.ndarray]): A dictionary containing homology types and their intervals.

    Returns:
        plt.Figure: The matplotlib figure containing the barcode diagram.
    """
    homologies = list(barcode.keys())
    np_lots = len(homologies)
    fig, axes = plt.subplots(np_lots, figsize=(15, min(10, 5 * np_lots)))

    if np_lots == 1:
        axes = [axes]

    for i, name in enumerate(homologies):
        axes[i].set_title(name)
        axes[i].set_ylim([-0.05, 1.05])
        bars = barcode[name]
        n = len(bars)
        for j, bar in enumerate(bars):
            axes[i].plot([bar[0], bar[1]], [j / n, j / n], color="black")
        axes[i].set_yticks([])

    plt.close(fig)
    return fig


def compute_data_barcode(data: torch.Tensor, hom_type: str, coefficient_type: str) -> Dict[str, np.ndarray]:
    """
    Computes a barcode for the given data using persistent homology.

    Args:
        data (torch.Tensor): The input data for barcode computation.
        hom_type (str): The type of homology to use ("standard", "sparse", "weak").
        coefficient_type (str): The coefficient field for homology computation.

    Returns:
        Dict[str, np.ndarray]: The computed barcode as a dictionary.

    Raises:
        ValueError: If an invalid hom_type is provided.
    """
    if hom_type == "standard":
        vr = VietorisRipsPersistence(
            homology_dimensions=[0],
            collapse_edges=True,
            coeff=int(coefficient_type),
        )
    elif hom_type == "sparse":
        vr = SparseRipsPersistence(homology_dimensions=[0], coeff=int(coefficient_type))
    elif hom_type == "weak":
        vr = WeakAlphaPersistence(
            homology_dimensions=[0],
            collapse_edges=True,
            coeff=int(coefficient_type),
        )
    else:
        raise ValueError('hom_type must be one of: "standard", "sparse", "weak"!')

    if data.ndim > 2:
        data = torch.nn.Flatten()(data)
    data = data.reshape(1, *data.shape)
    diagrams = vr.fit_transform(data)
    plot = vr.plot(diagrams)
    return _diagram_to_barcode(plot)


def compute_nn_barcode(
        model: torch.nn.Module,
        x: torch.Tensor,
        layer: str,
        hom_type: str,
        coefficient_type: str,
) -> Dict[str, np.ndarray]:
    """
    Computes a barcode for a specified layer in a neural network model.

    Args:
        model (torch.nn.Module): The neural network model.
        x (torch.Tensor): The input data for the model.
        layer (str): The layer to extract activations from.
        hom_type (str): The type of homology to use ("standard", "sparse", "weak").
        coefficient_type (str): The coefficient field for homology computation.

    Returns:
        Dict[str, np.ndarray]: The computed barcode as a dictionary.
    """
    activation = _get_activation(model, x, layer)
    return compute_data_barcode(activation, hom_type, coefficient_type)
