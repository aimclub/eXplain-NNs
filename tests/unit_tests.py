from collections import OrderedDict
import plotly
import numpy as np
import torch
import torch.nn as nn
# from eXNN.InnerNeuralTopology import api as topology_api
from eXNN.InnerNeuralViz import api as viz_api
from tests.test_utils import compare_values
# from eXNN.NetBayesianization import api as bayes_api


def test_check_random_input():
    shape = [5, 17, 81, 37]
    data = viz_api.get_random_input(shape)
    compare_values(type(data), torch.Tensor, 'Wrong result type')
    compare_values(torch.Size(shape), data.shape, 'Wrong result shape')

def _check_reduce_dim(mode):
    N = 20
    dim = 256
    data = torch.randn((N, dim))
    reduced_data = viz_api.ReduceDim(data, mode)
    compare_values(np.ndarray, type(reduced_data), 'Wrong result type')
    compare_values((N, 2), reduced_data.shape, 'Wrong result shape')

def test_reduce_dim_umap():
    _check_reduce_dim('umap')

def test_reduce_dim_pca():
    _check_reduce_dim('pca')

def test_visualization():
    N = 20
    dim = 256
    data = torch.randn((N, dim))

    model = nn.Sequential(
        OrderedDict([
            ('first_layer', nn.Linear(256, 128)), 
            ('second_layer', nn.Linear(128, 64)), 
            ('third_layer', nn.Linear(64, 10))
        ])
    )

    layers = ['second_layer', 'third_layer']
    res = viz_api.VisualizeNetSpace(model, 'umap', data, layers=layers)

    compare_values(dict, type(res), 'Wrong result type')
    compare_values(3, len(res), 'Wrong dictionary length')
    compare_values(set(['input'] + layers), set(res.keys()), 'Wrong dictionary keys')
    for key, plot in res.items():
        compare_values(plotly.graph_objs._figure.Figure, type(plot), f'Wrong value type for key {key}')