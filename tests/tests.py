import matplotlib
import numpy as np
import plotly
import torch

import eXNN.bayes as bayes_api
import eXNN.topology as topology_api
import eXNN.visualization as viz_api
import tests.test_utils as utils


def test_check_random_input():
    shape = [5, 17, 81, 37]
    data = viz_api.get_random_input(shape)
    utils.compare_values(type(data), torch.Tensor, "Wrong result type")
    utils.compare_values(torch.Size(shape), data.shape, "Wrong result shape")


def _check_reduce_dim(mode):
    data = utils.create_testing_data()
    reduced_data = viz_api.reduce_dim(data, mode)
    utils.compare_values(np.ndarray, type(reduced_data), "Wrong result type")
    utils.compare_values((len(data), 2), reduced_data.shape, "Wrong result shape")


def test_reduce_dim_umap():
    _check_reduce_dim("umap")


def test_reduce_dim_pca():
    _check_reduce_dim("pca")


def test_visualization_fcn():
    data = utils.create_testing_data()
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

def test_visualization_cnn():
    data = utils.create_testing_data(architecture='cnn')
    model = utils.create_testing_model(architecture='cnn')
    layers = ["first_layer", "second_layer", "avgpool", "flatten", "fc"]
    res = viz_api.visualize_layer_manifolds(model, "umap", data, layers=layers)

    utils.compare_values(dict, type(res), "Wrong result type")
    utils.compare_values(6, len(res), "Wrong dictionary length")
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
    data = torch.randn((20, 1, 256))
    labels = torch.randn((20))
    model = utils.create_testing_model('rnn')
    layers = ["second_layer", "third_layer"]
    res = viz_api.visualize_recurrent_layer_manifolds(model, "umap",
                                                      data, layers=layers, labels=labels)
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


def _test_bayes_prediction(mode: str, architecture='fcn'):
    params = {
        "basic": dict(mode="basic", p=0.5),
        "beta": dict(mode="beta", a=0.9, b=0.2),
        "gauss": dict(sigma=1e-2),
    }

    data = utils.create_testing_data(architecture=architecture)
    num_classes = 17
    model = utils.create_testing_model(architecture=architecture, num_classes=num_classes)
    n_iter = 7
    if mode != 'gauss':
        res = bayes_api.DropoutBayesianWrapper(model, **(params[mode])).predict(data, n_iter=n_iter)
    else:
        res = bayes_api.GaussianBayesianWrapper(model, **(params[mode])).predict(data,
                                                                                 n_iter=n_iter)

    utils.compare_values(dict, type(res), "Wrong result type")
    utils.compare_values(2, len(res), "Wrong dictionary length")
    utils.compare_values(set(["mean", "std"]), set(res.keys()), "Wrong dictionary keys")
    utils.compare_values(torch.Size([len(data), num_classes]), res["mean"].shape, "Wrong mean shape")
    utils.compare_values(torch.Size([len(data), num_classes]), res["std"].shape, "Wrong mean std")


def test_basic_bayes_wrapper():
    _test_bayes_prediction("basic")


def test_beta_bayes_wrapper():
    _test_bayes_prediction("beta")


def test_gauss_bayes_wrapper():
    _test_bayes_prediction("gauss")


def test_bayes_wrapper_cnn():
    _test_bayes_prediction("basic", architecture='cnn')


def test_data_barcode():
    data = utils.create_testing_data()
    res = topology_api.get_data_barcode(data, "standard", "3")
    utils.compare_values(dict, type(res), "Wrong result type")


def test_nn_barcodes():
    data = utils.create_testing_data()
    model = utils.create_testing_model()
    layers = ["second_layer", "third_layer"]
    res = topology_api.get_nn_barcodes(model, data, layers, "standard", "3")
    utils.compare_values(dict, type(res), "Wrong result type")
    utils.compare_values(2, len(res), "Wrong dictionary length")
    utils.compare_values(set(layers), set(res.keys()), "Wrong dictionary keys")
    for layer, barcode in res.items():
        utils.compare_values(
            dict,
            type(barcode),
            f"Wrong result type for key {layer}",
        )


def test_barcode_plot():
    data = utils.create_testing_data()
    barcode = topology_api.get_data_barcode(data, "standard", "3")
    plot = topology_api.plot_barcode(barcode)
    utils.compare_values(matplotlib.figure.Figure, type(plot), "Wrong result type")


def test_barcode_evaluate_all_metrics():
    data = utils.create_testing_data()
    barcode = topology_api.get_data_barcode(data, "standard", "3")
    result = topology_api.evaluate_barcode(barcode)
    utils.compare_values(dict, type(result), "Wrong result type")
    all_metric_names = [
        "h",
        "max_length",
        "mean_birth",
        "mean_death",
        "mean_length",
        "median_length",
        "normh",
        "ratio_2_1",
        "ratio_3_1",
        "snr",
        "stdev_birth",
        "stdev_death",
        "stdev_length",
        "sum_length",
    ]
    utils.compare_values(all_metric_names, sorted(result.keys()))
    for name, value in result.items():
        utils.compare_values(float, type(value), f"Wrong result type for metric {name}")


def test_barcode_evaluate_one_metric():
    data = utils.create_testing_data()
    barcode = topology_api.get_data_barcode(data, "standard", "3")
    result = topology_api.evaluate_barcode(barcode, metric_name="mean_length")
    utils.compare_values(float, type(result), "Wrong result type")
