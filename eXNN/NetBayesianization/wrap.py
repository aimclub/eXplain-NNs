from argparse import ArgumentError
import torch
import torch.nn as nn
from typing import Optional

def _check_p(p):
    assert p is not None, "Param `p` must not be None"
    assert p >= 0, "Param `p` must be >= 0"
    assert p <= 1, "Param `p` must be <= 1"

def create_bayesian_wrapper(model: torch.nn.Module,
                 mode: str,
                 p: Optional[float]=None,
                 a: Optional[float]=None,
                 b: Optional[float]=None) -> torch.nn.Module:
    if mode == 'basic':
        _check_p(p)
        new_layers = []
        for l in model.children():
            new_layers.append(l)
            if type(l).__name__ in ['Linear']: # add conv layers
                new_layers.append(nn.Dropout(p))
        return nn.Sequential(*new_layers)

    elif mode == 'beta':
        raise NotImplementedError("Mode `beta` for bayesianization is not implemented")
    else:
        raise ValueError(f"Unsupported mode `{mode}`")