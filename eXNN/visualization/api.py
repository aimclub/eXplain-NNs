import math
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import torch
import umap
from gtda.time_series import TakensEmbedding
from sklearn.decomposition import PCA
from torchvision.models.feature_extraction import create_feature_extractor

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
    out_dim=2,
    neighbors=20,
) -> np.ndarray:
    """This function reduces data dimensionality to 2 dimensions.

    Args:
        data (torch.Tensor): input data of shape NxC1x...xCk,
            where N is the number of data points,
            C1,...,Ck are dimensions of each data point
        mode (str): dimensionality reduction mode (`umap` or `pca`)
        out_dim (int): dimension of output, 3 by default
        neighbors (int): n_neighbors of umap method

    Raises:
        ValueError: returned if unsupported mode is provided

    Returns:
        np.ndarray: data projected on a 2d space, of shape Nx2
    """

    data = data.detach().cpu().numpy().reshape((len(data), -1))
    if mode == "pca":
        return PCA(n_components=out_dim).fit_transform(data)
    elif mode == "umap":
        return umap.UMAP(n_components=out_dim, n_neighbors=neighbors).fit_transform(data)
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
    arr_reducer=1,
    renderer='browser',
    heatmap: Optional[str] = True,
    layers: Optional[List[str]] = None,
    labels: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
) -> Dict[str, plotly.graph_objs.Figure]:
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
        arr_reducer (int): strips the output array of some data, leaving only each n_th
        renderer (str): plotly renderer for image,
            Available renderers:
                ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
                 'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
                 'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
                 'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
                 'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']
        layers (Optional[List[str]], optional): list of layers for visualization.
            Defaults to None. If None, visualization for all layers is performed
        labels (Optional[torch.Tensor], optional): data labels (colors).
            Defaults to None. If None, all points are visualized with the same color
        chunk_size (Optional[int], optional): batch size for data processing.
            Defaults to None. If None, all data is processed in one batch

    Returns:
        Dict[str, plotly.graph_objs.Figure]: dictionary with latent
            representations visualization for each layer
    """
    model2 = create_feature_extractor(model, return_nodes=layers)
    labels = labels.detach().numpy()
    emb_viz = {}
    for layer in layers:
        if torch.is_tensor(model2(data)[layer]):
            layer_output = model2(data)[layer].cpu().detach().numpy()
        else:
            layer_output = model2(data)[layer][0].cpu().detach().numpy()
        if stride_mode == 'dimensional':
            stride = layer_output.shape[layer_output.ndim - 1]
        else:
            stride = stride_mode
        if layer_output.ndim > 2:
            embedder = TakensEmbedding(time_delay=time_delay, dimension=embedding_dim,
                                       stride=stride)
            emb_res = embedder.fit_transform(layer_output)
        else:
            embedder = TakensEmbedding(time_delay=time_delay, dimension=embedding_dim,
                                       stride=stride)
            emb_res = embedder.fit_transform(layer_output.reshape(layer_output.shape[0],
                                                                  1, layer_output.shape[1]))
            emb_res = emb_res.reshape(emb_res.shape[0], 1, -1)
        emb_res = torch.from_numpy(emb_res)
        reducing_output = reduce_dim(data=emb_res[:, 0, :], mode=mode,
                                     out_dim=out_dim, neighbors=neighbors)
        df = pd.DataFrame(reducing_output)
        if labels.ndim == 1:
            df["category"] = labels.astype(str)
        else:
            df["category"] = np.where(labels == 1)[1].astype(str)
        df = df.iloc[::arr_reducer, :]
        if heatmap is True:
            labels_noncat = labels
            if labels_noncat[0].shape[0] > 1:
                labels_noncat = np.argmax(labels_noncat, axis=1)
            center = np.zeros((len(np.unique(labels_noncat)), 3))
            med_dist = np.zeros((len(np.unique(labels_noncat)), len(np.unique(labels_noncat))))
            for i in range(len(np.unique(labels_noncat))):
                center[i] = np.mean(reducing_output[np.where(labels_noncat == np.unique(
                    labels_noncat)[i])], axis=0)
                for j in range(len(np.unique(labels_noncat))):
                    med_dist[i][j] = math.log(np.square(1 / np.mean(reducing_output[np.where(
                        labels_noncat == np.unique(labels_noncat)[j])] - center[i])))
        emb_out = px.scatter_3d(df, x=0, y=1, z=2, color="category")
        emb_out.update_traces(marker=dict(size=4))
        emb_out.update_layout(
            autosize=False,
            width=1000,
            height=1000)
        emb_out.show(renderer="colab")
        emb_viz[layer] = emb_out
    return emb_viz


def get_random_input(dims: List[int]) -> torch.Tensor:
    """This function generates uniformly distributed tensor of given shape.

    Args:
        dims (List[int]): required data shape

    Returns:
        torch.Tensor: uniformly distributed tensor of given shape
    """
    return torch.rand(size=dims)
