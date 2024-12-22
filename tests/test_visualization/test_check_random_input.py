import torch

import eXNN.visualization as viz_api
import tests.utils.test_utils as utils


def test_check_random_input():
    """
    Test generating random input tensors with a specific shape.

    Verifies:
        - Output type is a torch.Tensor.
        - Output shape matches the specified shape.
    """
    shape = [5, 17, 81, 37]
    data = viz_api.get_random_input(shape)
    utils.compare_values(type(data), torch.Tensor, "Wrong result type")
    utils.compare_values(torch.Size(shape), data.shape, "Wrong result shape")
