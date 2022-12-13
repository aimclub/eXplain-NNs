import torch
from typing import List
from eXNN.InnerNeuralTopology.homologies import InnerNetspaceHomologies, _ComputeBarcode

def ComputeBarcode(data: torch.Tensor,
                   hom_type: str,
                   coefs_type: str):
    return _ComputeBarcode(data, hom_type, coefs_type)

def NetworkHomologies(model: torch.nn.Module,
                      data: torch.Tensor,
                      layers: List[str],
                      hom_type: str,
                      coefs_type: str):
    res = {}
    for layer in layers:
        res[layer] = InnerNetspaceHomologies(model, data, layer, hom_type, coefs_type)
    return res
