import matplotlib
import plotly
import torch

import eXNN.visualization as viz_api
import tests.utils.test_utils as utils


def test_visualization():
    """
    Test visualization of layer manifolds using UMAP.

    Verifies:
        - Output is a dictionary.
        - Dictionary has the correct keys corresponding to the input and specified layers.
        - Each dictionary value is a matplotlib.figure.Figure.
    """

    n, dim, data = utils.create_testing_data()
    model = utils.create_testing_model()
    layers = ["second_layer", "third_layer"]
    res = viz_api.visualize_layer_manifolds(model, "umap", data, layers=layers)

    utils.compare_values(dict, type(res), "Wrong result type")
    utils.compare_values(3, len(res), "Wrong dictionary length")
    utils.compare_values(
        set(["input"] + layers),
        set(res.keys()),
        "Wrong dictionary keys",
    )

    for key, plot in res.items():
        utils.compare_values(
            matplotlib.figure.Figure,
            type(plot),
            f"Wrong value type for key {key}",
        )


def test_embed_visualization():
    """
    Test visualization of recurrent layer manifolds with embeddings.

    Verifies:
        - Output is a dictionary.
        - Dictionary keys match the specified layers.
        - Each dictionary value is a plotly.graph_objs.Figure.
    """
    data = torch.randn((20, 1, 256))
    labels = torch.randn(20)
    model = utils.create_testing_model_lstm()
    layers = ["second_layer", "third_layer"]

    res = viz_api.visualize_recurrent_layer_manifolds(model, "umap", data, layers=layers,
                                                      labels=labels)

    utils.compare_values(dict, type(res), "Wrong result type")
    utils.compare_values(2, len(res), "Wrong dictionary length")
    utils.compare_values(
        set(layers),
        set(res.keys()),
        "Wrong dictionary keys",
    )
    for key, plot in res.items():
        utils.compare_values(
            plotly.graph_objs.Figure,
            type(plot),
            f"Wrong value type for key {key}",
        )
