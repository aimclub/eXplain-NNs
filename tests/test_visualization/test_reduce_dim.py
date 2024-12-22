import numpy as np

import eXNN.visualization as viz_api
import tests.utils.test_utils as utils


def _check_reduce_dim(mode):
    """
    Helper function to test dimensionality reduction with a given mode.

    Args:
        mode (str): Dimensionality reduction mode (e.g., "umap", "pca").

    Verifies:
        - Output type is a numpy.ndarray.
        - Output shape is (n_samples, 2).
    """
    n, dim, data = utils.create_testing_data()
    reduced_data = viz_api.reduce_dim(data, mode)
    utils.compare_values(np.ndarray, type(reduced_data), "Wrong result type")
    utils.compare_values((n, 2), reduced_data.shape, "Wrong result shape")


def test_reduce_dim_umap():
    """
    Test dimensionality reduction using UMAP.

    Uses:
        _check_reduce_dim with mode "umap".
    """
    _check_reduce_dim("umap")


def test_reduce_dim_pca():
    """
    Test dimensionality reduction using PCA.

    Uses:
        _check_reduce_dim with mode "pca".
    """
    _check_reduce_dim("pca")

