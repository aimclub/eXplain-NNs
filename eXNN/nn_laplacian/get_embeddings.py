import torch
import numpy as np 
from scipy.sparse import csc_matrix, csgraph


def get_activation(model: torch.nn.Module, x: torch.Tensor, layer: str):
    '''
    '''
    # Check if layer is model's module
    err_missing_layer = f"Layer '{layer}' is missing in model's state dict!"
    assert hasattr(model, layer), err_missing_layer
    
    activation = {}
    def store_output(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Attach forward hook to layer
    layer_obj = getattr(model, layer)
    h1 = layer_obj.register_forward_hook(store_output(layer))
    
    try:
        model.forward(x)
        h1.remove()
    except Exception as error:
        assert 0, 'Error occured while registering activations: ' + repr(error)
        
    return activation[layer]


def get_pairwise_distance(emb: torch.Tensor):
    
    err_dim = f"dimension must be either (N, D) or (N, C, W, H), got {emb.dim()}"
    assert emb.dim() in [2, 4], err_dim
    
    if emb.dim() == 4:
        emb = torch.flatten(emb, start_dim = 1)
    
    dist_matrix = torch.cdist(emb, emb)
    return dist_matrix