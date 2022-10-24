import torch
from typing import List, Optional
from sklearn.decomposition import PCA
import umap
import plotly.express as px
from eXNN.InnerNeuralViz.hook import get_hook

def _plot(embedding):
    return px.scatter(embedding[:,0], embedding[:,1])


def ReduceDim(data: torch.Tensor,
              mode: str):
    data = data.detach().cpu().numpy().reshape((len(data), -1))
    if mode == 'pca':
        return PCA(n_components=2).fit_transform(data)
    elif mode == 'umap':
        return umap.UMAP().fit_transform(data)
    else:
        raise ValueError(f'Unsupported mode: `{mode}`')

def VisualizeNetSpace(model: torch.nn.Module,
                      mode: str,
                      data: torch.Tensor,
                      layers: Optional[List[str]]=None):
    if layers is None:
        layers = [_[0] for _ in model.named_children()]
    hooks = {layer: get_hook(model, layer) for layer in layers}
    with torch.no_grad():
        out = model(data)
    visualizations = {'input': _plot(ReduceDim(data, mode))}
    for layer in layers:
        visualizations[layer] = _plot(ReduceDim(hooks[layer].fwd, mode))
    return visualizations

def get_random_input(dims: List[int]):
    """Generates random data of dims=`dims` drawn from uniform distribution"""
    return torch.rand(size=dims)
