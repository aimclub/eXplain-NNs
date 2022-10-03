import torch
from typing import List


def ReduceDim(data: torch.Tensor,
              mode: str):
    raise NotImplementedError()

def VisualizeNetSpace(model: torch.nn.Module,
                      data: torch.Tensor,
                      layers: List[str],
                      mode: str):
    raise NotImplementedError()