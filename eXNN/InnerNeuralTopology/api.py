import torch
import matplotlib.pyplot as plt
from typing import List, Dict
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence, SparseRipsPersistence
from gtda.plotting import plot_diagram


def GetActivation(model: torch.nn.Module,
                   x: torch.Tensor,
                   layer: str):
    activation = {}

    def getActivation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    h1 = getattr(model, layer).register_forward_hook(getActivation(layer))
    model.forward(x)
    h1.remove()
    return activation[layer]


def diagram_to_barcode(plot):
    data = plot['data']
    homologies = {}
    for h in data:
        if h['name'] is None:
            continue
        homologies[h['name']] = list(zip(h['x'], h['y']))

    for h in homologies.keys():
        homologies[h] = sorted(homologies[h], key=lambda x: x[0])
    return homologies


def plot_barcode(barcode: Dict):
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
            ax[i].plot([bar[0], bar[1]], [j / n, j / n], color='black')
        labels = ["" for _ in range(len(ax[i].get_yticklabels()))]
        ax[i].set_yticklabels(labels)

    if nplots == 1:
        ax = ax[0]
    plt.close(fig)
    return fig

def InnerNetspaceHomologies(model: torch.nn.Module,
                          x: torch.Tensor,
                          layer: str,
                          dimensions: List[int] = [0, 1],
                          make_barplot: bool = True,
                          rm_empty: bool = True):
    act = GetActivation(model, x, layer)
    act = act.reshape(1, *act.shape)
    # Dimensions must not be outside layer dimensionality
    N = act.shape[-1]
    dimensions = [i if i >= 0 else N + i for i in dimensions]
    dimensions = [i for i in dimensions if ((i >= 0) and (i < N))]
    dimensions = list(set(dimensions))
    VR = VietorisRipsPersistence(homology_dimensions=dimensions,
                                 collapse_edges = True)
    diagrams = VR.fit_transform(act)
    plot = VR.plot(diagrams)
    if make_barplot:
        barcode = diagram_to_barcode(plot)
        if rm_empty:
            barcode = {key: val for key, val in barcode.items() if len(val) > 0}
        return plot_barcode(barcode)
    else:
        return plot

def ComputeBarcode(data: torch.Tensor,
                   hom_type: str,
                   coefs_type: str):
    raise NotImplementedError()

def NetworkHomologies(model: torch.nn.Module,
                      data: torch.Tensor,
                      layers: List[str],
                      hom_type: str,
                      coefs_type: str):
    raise NotImplementedError()