import torch

import eXNN.bayes as bayes_api
import tests.utils.test_utils as utils


def _test_bayes_prediction(mode: str):
    """
    Helper function to test Bayesian wrappers with different configurations.

    Args:
        mode (str): Bayesian wrapper mode ("basic", "beta", "gauss").

    Verifies:
        - Output is a dictionary with keys "mean" and "std".
        - Shapes of "mean" and "std" match the expected shape.
    """
    params = {
        "basic": dict(mode="basic", p=0.5),
        "beta": dict(mode="beta", a=0.9, b=0.2),
        "gauss": dict(sigma=1e-2),
    }

    n, dim, data = utils.create_testing_data()
    num_classes = 17
    model = utils.create_testing_model(num_classes=num_classes)
    n_iter = 7

    if mode != 'gauss':
        res = bayes_api.DropoutBayesianWrapper(model, **(params[mode])).predict(data, n_iter=n_iter)
    else:
        res = bayes_api.GaussianBayesianWrapper(model, **(params[mode])).predict(data, n_iter=n_iter)

    utils.compare_values(dict, type(res), "Wrong result type")
    utils.compare_values(2, len(res), "Wrong dictionary length")
    utils.compare_values({"mean", "std"}, set(res.keys()), "Wrong dictionary keys")
    utils.compare_values(torch.Size([n, num_classes]), res["mean"].shape, "Wrong mean shape")
    utils.compare_values(torch.Size([n, num_classes]), res["std"].shape, "Wrong mean std")


def test_basic_bayes_wrapper():
    """
    Test the basic Bayesian wrapper.
    """
    _test_bayes_prediction("basic")


def test_beta_bayes_wrapper():
    """
    Test the beta Bayesian wrapper.
    """
    _test_bayes_prediction("beta")


def test_gauss_bayes_wrapper():
    """
    Test the Gaussian Bayesian wrapper.
    """
    _test_bayes_prediction("gauss")

