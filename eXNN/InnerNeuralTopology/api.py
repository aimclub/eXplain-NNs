import torch
from typing import List


def ComputeBarcode(data: torch.Tensor,
                   hom_type: str,
                   coefs_type: str):
    raise NotImplementedError()

def NetworkHomologies(model: torch.nn.Module,
                      data: torch.Tensor,
                      layers: List[str],
                      hom_type: str,
                      coefs_type: str):
    raise NotImplementedError()