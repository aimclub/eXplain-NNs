from typing import Union
import torch
from torch import Tensor
from torch.nn import Conv2d, Parameter, Module
from torch.nn.common_types import _size_2_t


class DecomposedConv2d(Conv2d):
    """Extends the Conv2d layer by implementing the singular value decomposition of
    the weight matrix.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        decomposing: bool = True,
        decomposing_mode: str = "channel",
        device=None,
        dtype=None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        n, c, w, h = self.weight.size()
        self.decomposing_modes_dict = {
            "channel": (n, c * w * h),
            "spatial": (n * w, c * h),
        }

        if decomposing:
            self.decompose(decomposing_mode)
        else:
            self.U = None
            self.S = None
            self.Vh = None
            self.decomposing = False

    def decompose(self, decomposing_mode: str) -> None:
        """Decompose the weight matrix in singular value decomposition."""

        if decomposing_mode not in self.decomposing_modes_dict.keys():
            raise ValueError(
                "decomposing_mode must be one of {}, but got decomposing_mode='{}'".format(
                    self.decomposing_modes_dict.keys(), decomposing_mode
                )
            )
        W = self.weight.view(self.decomposing_modes_dict[decomposing_mode])
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        self.U = Parameter(U)
        self.S = Parameter(S)
        self.Vh = Parameter(Vh)
        self.register_parameter("weight", None)
        self.decomposing = True

    def compose(self) -> None:
        """Compose the weight matrix from singular value decomposition."""

        W = self.U @ torch.diag(self.S) @ self.Vh
        self.weight = Parameter(
            W.view(
                self.out_channels, self.in_channels // self.groups, *self.kernel_size
            )
        )

        self.register_parameter("U", None)
        self.register_parameter("S", None)
        self.register_parameter("Vh", None)
        self.decomposing = False

    def forward(self, input: Tensor) -> Tensor:

        if self.decomposing:
            W = self.U @ torch.diag(self.S) @ self.Vh
            return self._conv_forward(
                input,
                W.view(
                    self.out_channels,
                    self.in_channels // self.groups,
                    *self.kernel_size
                ),
                self.bias,
            )
        else:
            return self._conv_forward(input, self.weight, self.bias)

    def set_U_S_Vh(self, u: Tensor, s: Tensor, vh: Tensor) -> None:
        """Update U, S, Vh matrices."""

        assert self.decomposing, "for setting U, S and Vh, the model must be decomposed"
        self.U = Parameter(u)
        self.S = Parameter(s)
        self.Vh = Parameter(vh)

def energy_threshold_pruning(conv: DecomposedConv2d, energy_threshold: float) -> None:
    """Prune the weight matrices to the energy_threshold (in-place)."""
    assert conv.decomposing, "for pruning, the model must be decomposed"
    S, indices = conv.S.sort()
    U = conv.U[:, indices]
    Vh = conv.Vh[indices, :]
    sum = (S ** 2).sum()
    threshold = energy_threshold * sum
    for i, s in enumerate(S):
        sum -= s ** 2
        if sum < threshold:
            conv.set_U_S_Vh(U[:, i:].clone(), S[i:].clone(), Vh[i:, :].clone())
            break


def decompose_module(model: Module, decomposing_mode: str = "channel") -> None:
    """Replace Conv2d layers with DecomposedConv2d layers in module (in-place)."""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            decompose_module(module, decomposing_mode=decomposing_mode)

        if isinstance(module, Conv2d):
            new_module = DecomposedConv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
                decomposing=False,
            )
            new_module.load_state_dict(module.state_dict())
            new_module.decompose(decomposing_mode=decomposing_mode)
            setattr(model, name, new_module)

def prune_model(model, energy_threshold) -> None:
    """Prune the model weights to the energy_threshold."""
    for module in model.modules():
        if isinstance(module, DecomposedConv2d):
            energy_threshold_pruning(conv=module, energy_threshold=energy_threshold)

def number_of_params(model) -> int:
    """Return number of model parameters."""
    return sum(p.numel() for p in model.parameters())