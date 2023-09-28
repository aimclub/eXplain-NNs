import matplotlib
import numpy as np
import torch

import eXNN.topology as topology_api
import eXNN.visualization as viz_api
import eXNN.bayes as bayes_api
import tests.test_utils as utils


def test_check_random_input():
    shape = [5, 17, 81, 37]
    data = viz_api.get_random_input(shape)
    utils.compare_values(type(data), torch.Tensor, "Wrong result type")
    utils.compare_values(torch.Size(shape), data.shape, "Wrong result shape")


def _check_reduce_dim(mode):
    N, dim, data = utils.create_testing_data()
    reduced_data = viz_api.ReduceDim(data, mode)
    utils.compare_values(np.ndarray, type(reduced_data), "Wrong result type")
    utils.compare_values((N, 2), reduced_data.shape, "Wrong result shape")


def test_reduce_dim_umap():
    _check_reduce_dim("umap")


def test_reduce_dim_pca():
    _check_reduce_dim("pca")


def test_visualization():
    N, dim, data = utils.create_testing_data()
    model = utils.create_testing_model()
    layers = ["second_layer", "third_layer"]
    res = viz_api.VisualizeNetSpace(model, "umap", data, layers=layers)

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


def _test_bayes_prediction(use_wrapper: bool, mode: str):
    params = {"basic": dict(mode="basic", p=0.5), "beta": dict(mode="beta", a=0.9, b=0.2)}

    N, dim, data = utils.create_testing_data()
    model = utils.create_testing_model()
    n_iter = 10
    if use_wrapper:
        res = bayes_api.BasicBayesianWrapper(model, **(params[mode])).predict(data, n_iter=n_iter)
    else:
        res = bayes_api.BasicBayesianPrediction(data, model, n_iter=n_iter, **(params[mode]))

    utils.compare_values(dict, type(res), "Wrong result type")
    utils.compare_values(2, len(res), "Wrong dictionary length")
    utils.compare_values(set(["mean", "std"]), set(res.keys()), "Wrong dictionary keys")
    utils.compare_values(torch.Size([N, n_iter]), res["mean"].shape, "Wrong mean shape")
    utils.compare_values(torch.Size([N, n_iter]), res["std"].shape, "Wrong mean std")


def test_basic_bayes_prediction():
    _test_bayes_prediction(False, "basic")


def test_beta_bayes_prediction():
    _test_bayes_prediction(False, "beta")


def test_basic_bayes_wrapper():
    _test_bayes_prediction(True, "basic")


def test_beta_bayes_wrapper():
    _test_bayes_prediction(True, "beta")


def test_barcodes():
    N, dim, data = utils.create_testing_data()
    res = topology_api.ComputeBarcode(data, "standard", "3")
    utils.compare_values(matplotlib.figure.Figure, type(res), "Wrong result type")


def test_homologies():
    N, dim, data = utils.create_testing_data()
    model = utils.create_testing_model()
    layers = ["second_layer", "third_layer"]
    res = topology_api.NetworkHomologies(model, data, layers, "standard", "3")
    utils.compare_values(dict, type(res), "Wrong result type")
    utils.compare_values(2, len(res), "Wrong dictionary length")
    utils.compare_values(set(layers), set(res.keys()), "Wrong dictionary keys")
    for layer, plot in res.items():
        utils.compare_values(
            matplotlib.figure.Figure,
            type(plot),
            f"Wrong result type for key {layer}",
        )
