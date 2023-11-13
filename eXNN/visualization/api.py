import math
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import numpy as np
import torch
import umap
import pandas as pd
from sklearn.decomposition import PCA
from gtda.time_series import TakensEmbedding

from eXNN.visualization.hook import get_hook


def _plot(embedding, labels):
    fig, ax = plt.subplots()
    ax.scatter(x=embedding[:, 0], y=embedding[:, 1], c=labels)
    ax.axis("Off")
    plt.close()
    return fig


def reduce_dim(
    data: torch.Tensor,
    mode: str,
) -> np.ndarray:
    """This function reduces data dimensionality to 2 dimensions.

    Args:
        data (torch.Tensor): input data of shape NxC1x...xCk,
            where N is the number of data points,
            C1,...,Ck are dimensions of each data point
        mode (str): dimensionality reduction mode (`umap` or `pca`)

    Raises:
        ValueError: returned if unsupported mode is provided

    Returns:
        np.ndarray: data projected on a 2d space, of shape Nx2
    """

    data = data.detach().cpu().numpy().reshape((len(data), -1))
    if mode == "pca":
        return PCA(n_components=2).fit_transform(data)
    elif mode == "umap":
        return umap.UMAP().fit_transform(data)
    else:
        raise ValueError(f"Unsupported mode: `{mode}`")


def visualize_layer_manifolds(
    model: torch.nn.Module,
    mode: str,
    data: torch.Tensor,
    layers: Optional[List[str]] = None,
    labels: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
) -> Dict[str, matplotlib.figure.Figure]:
    """This function visulizes data latent representations on neural network layers.

    Args:
        model (torch.nn.Module): neural network
        mode (str): dimensionality reduction mode (`umap` or `pca`)
        data (torch.Tensor): input data of shape NxC1x...xCk,
            where N is the number of data points,
            C1,...,Ck are dimensions of each data point
        layers (Optional[List[str]], optional): list of layers for visualization.
            Defaults to None. If None, visualization for all layers is performed
        labels (Optional[torch.Tensor], optional): data labels (colors).
            Defaults to None. If None, all points are visualized with the same color
        chunk_size (Optional[int], optional): batch size for data processing.
            Defaults to None. If None, all data is processed in one batch

    Returns:
        Dict[str, matplotlib.figure.Figure]: dictionary with latent
            representations visualization for each layer
    """

    if layers is None:
        layers = [_[0] for _ in model.named_children()]
    if labels is not None:
        labels = labels.detach().cpu().numpy()
    hooks = {layer: get_hook(model, layer) for layer in layers}
    if chunk_size is None:
        with torch.no_grad():
            _ = model(data)
        visualizations = {"input": _plot(reduce_dim(data, mode), labels)}
        for layer in layers:
            visualizations[layer] = _plot(reduce_dim(hooks[layer].fwd, mode), labels)
        return visualizations
    else:
        representations = {layer: [] for layer in layers}
        for i in range(math.ceil(len(data) / chunk_size)):
            with torch.no_grad():
                _ = model(data[(i * chunk_size) : ((i + 1) * chunk_size)])
            for layer in layers:
                representations[layer].append(hooks[layer].fwd.detach().cpu())
        visualizations = {"input": _plot(reduce_dim(data, mode), labels)}
        for layer in layers:
            layer_reprs = torch.cat(representations[layer], dim=0)
            visualizations[layer] = _plot(reduce_dim(layer_reprs, mode), labels)
        return visualizations


def visualize_recurrent_layer_manifolds(
    model: torch.nn.Module,
    mode: str,
    data: torch.Tensor,
    neighbors=20,
    time_delay=1,
    embedding_dim=10,
    stride_mode='dimensional',
    out_dim=3,
    renderer='browser',
    layers: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
) -> Dict[str, plotly.graph_objs._figure.Figure]:
    """This function visulizes data latent representations on neural network layers.

    Args:
        model (torch.nn.Module): neural network
        mode (str): dimensionality reduction mode (`umap` or `pca`)
        neighbors (int): n_neighbors of umap method
        data (torch.Tensor): input data of shape NxC1x...xCk,
            where N is the number of data points,
            C1,...,Ck are dimensions of each data point
        time_delay (int): td between two consecutive values for constructing one embedded point
        embedding_dim (int): number of resulting features
        stride_mode ('dimensional' or str): stride duration between two consecutive embedded points,
            'dimensional' makes 'stride' equal to layer dimension
        out_dim (int): dimension of output, 3 by default
        renderer (str): plotly renderer for image,
            Available renderers:
                ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
                 'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
                 'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
                 'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
                 'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']
        layers (Optional[List[str]], optional): list of layers for visualization.
            Defaults to None. If None, visualization for all layers is performed
        chunk_size (Optional[int], optional): batch size for data processing.
            Defaults to None. If None, all data is processed in one batch

    Returns:
        Dict[str, plotly.graph_objs._figure.Figure]: dictionary with latent
            representations visualization for each layer
    """
    if layers is None:
        layers = [_[0] for _ in model.named_children()]
    layer_output = {layer: get_hook(model, layer) for layer in layers}
    if stride_mode == 'dimensional':
        stride = layer_output.shape[layer_output.ndim - 1]
    else:
        stride = stride_mode
    if layer_output.ndim > 2:
        embedder = TakensEmbedding(time_delay=time_delay, dimension=10, stride=stride)
        emb_res = embedder.fit_transform(layer_output[:, 0, :].reshape(1,-1))
    else:
        embedder = TakensEmbedding(time_delay=time_delay, dimension=10, stride=stride)
        emb_res = embedder.fit_transform(layer_output.reshape(1, -1))
    if mode.lower() == 'umap':
        umapred = umap.UMAP(n_components=3, n_neighbors=neighbors)
        reducing_output = umapred.fit_transform(emb_res[0, :, :])
    if mode.lower() == 'pca':
        PCA_out = PCA(n_components=3)
        reducing_output = PCA_out.fit_transform(emb_res[0, :, :])
    df = pd.DataFrame(reducing_output)
    df["category"] = y_train.astype(str)
    df = df.iloc[::4, :]
    emb_out = px.scatter_3d(df, x=0, y=1, z=2, color='category')
    emb_out.update_traces(marker=dict(size=4))
    emb_out.update_layout(
        autosize=False,
        width=1000,
        height=1000)
    emb_out.show(renderer="colab")


def get_random_input(dims: List[int]) -> torch.Tensor:
    """This function generates uniformly distributed tensor of given shape.

    Args:
        dims (List[int]): required data shape

    Returns:
        torch.Tensor: uniformly distributed tensor of given shape
    """
    return torch.rand(size=dims)
