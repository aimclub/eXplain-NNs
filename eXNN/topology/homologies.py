from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import numpy as np
from gtda.homology import (
    SparseRipsPersistence,
    VietorisRipsPersistence,
    WeakAlphaPersistence,
)


def _get_activation(model: torch.nn.Module, x: torch.Tensor, layer: str):
    activation = {}

    def store_output(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    h1 = getattr(model, layer).register_forward_hook(store_output(layer))
    model.forward(x)
    h1.remove()
    return activation[layer]


def _diagram_to_barcode(plot):
    data = plot["data"]
    homologies = {}
    for h in data:
        if h["name"] is None:
            continue
        homologies[h["name"]] = list(zip(h["x"], h["y"]))

    for h in homologies.keys():
        homologies[h] = sorted(homologies[h], key=lambda x: x[0])
    return homologies


def plot_barcode(barcode: Dict[str, np.ndarray]):
    homologies = list(barcode.keys())
    nplots = len(homologies)
    fig, ax = plt.subplots(nplots, figsize=(15, min(10, 5 * nplots)))

    if nplots == 1:
        ax = [ax]

    for i in range(nplots):
        name = homologies[i]
        ax[i].set_title(name)
        ax[i].set_ylim([-0.05, 1.05])
        bars = barcode[name]
        n = len(bars)
        for j in range(n):
            bar = bars[j]
            ax[i].plot([bar[0], bar[1]], [j / n, j / n], color="black")
        labels = ["" for _ in range(len(ax[i].get_yticklabels()))]
        ax[i].set_yticks(ax[i].get_yticks())
        ax[i].set_yticklabels(labels)

    if nplots == 1:
        ax = ax[0]
    plt.close(fig)
    return fig


def compute_data_barcode(data: torch.Tensor, hom_type: str, coefs_type: str):
    if hom_type == "standard":
        VR = VietorisRipsPersistence(
            homology_dimensions=[0],
            collapse_edges=True,
            coeff=int(coefs_type),
        )
    elif hom_type == "sparse":
        VR = SparseRipsPersistence(homology_dimensions=[0], coeff=int(coefs_type))
    elif hom_type == "weak":
        VR = WeakAlphaPersistence(
            homology_dimensions=[0],
            collapse_edges=True,
            coeff=int(coefs_type),
        )
    else:
        raise Exception('hom_type must be one of: "standard", "sparse", "weak"!')

    if len(data.shape) > 2:
        data = torch.nn.Flatten()(data)
    data = data.reshape(1, *data.shape)
    diagrams = VR.fit_transform(data)
    plot = VR.plot(diagrams)
    return _diagram_to_barcode(plot)


def compute_nn_barcode(
    model: torch.nn.Module,
    x: torch.Tensor,
    layer: str,
    hom_type: str,
    coefs_type: str,
):
    act = _get_activation(model, x, layer)
    return compute_data_barcode(act, hom_type, coefs_type)


# def get_homologies_experimental(
#     model: torch.nn.Module,
#     x: torch.Tensor,
#     layer: str,
#     dimensions: Optional[List[int]] = None,
#     make_barplot: bool = True,
#     rm_empty: bool = True,
# ):
#     act = _get_activation(model, x, layer)
#     act = act.reshape(1, *act.shape)
#     # Dimensions must not be outside layer dimensionality
#     N = act.shape[-1]
#     dimensions = dimensions if dimensions is not None else []
#     dimensions = [i if i >= 0 else N + i for i in dimensions]
#     dimensions = [i for i in dimensions if ((i >= 0) and (i < N))]
#     dimensions = list(set(dimensions))
#     VR = VietorisRipsPersistence(homology_dimensions=dimensions, collapse_edges=True)
#     diagrams = VR.fit_transform(act)
#     plot = VR.plot(diagrams)
#     if make_barplot:
#         barcode = _diagram_to_barcode(plot)
#         if rm_empty:
#             barcode = {key: val for key, val in barcode.items() if len(val) > 0}
#         return _plot_barcode(barcode)
#     else:
#         return plot
