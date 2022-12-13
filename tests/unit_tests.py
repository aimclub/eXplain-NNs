import torch
# from eXNN.InnerNeuralTopology import api as topology_api
from eXNN.InnerNeuralViz import api as viz_api
# from eXNN.NetBayesianization import api as bayes_api


def test_check_random_input():
    data = viz_api.get_random_input([5, 17, 81, 37])
    assert type(data) == torch.Tensor, f"Wrong returned type: expected {torch.Tensor}, got {type(data)}"
    assert data.shape == torch.Size([5, 17, 81, 37]), f"Wrong tensor shape: expected {torch.Size([5, 17, 81, 37])}, got {data.shape}"