import torch
from typing import List
from sklearn.decomposition import PCA
import umap
import plotly.express as px
from eXNN.InnerNeuralViz.hook import get_hook

def _plot(embedding):
    return px.scatter(embedding[:,0], embedding[:,1])


def ReduceDim(data: torch.Tensor,
              mode: str):
    data = data.detach().cpu().numpy()
    if mode == 'pca':
        return PCA(n_components=2).fit_transform(data)
    elif mode == 'umap':
        return umap.UMAP().fit_transform(data)
    else:
        raise ValueError(f'Unsupported mode: `{mode}`')

def VisualizeNetSpace(model: torch.nn.Module,
                      data: torch.Tensor,
                      layers: List[str],
                      mode: str):
    hooks = {layer: get_hook(model, layer) for layer in layers}
    with torch.no_grad():
        out = model(data)
    visualizations = {'input': _plot(ReduceDim(data, mode))}
    for layer in layers:
        visualizations[layer] = _plot(ReduceDim(hooks[layer].fwd, mode))
    return visualizations