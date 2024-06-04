# NOTE: adapted from https://github.com/PeizhuoLi/ganimator

import logging
import math
import typing

import torch

log = logging.getLogger(__name__)

__all__ = [
    "SkeletonConvolution",
]


class SkeletonConvolution(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        padding: int = 0,
        padding_mode: str = "zeros",
        neighbors: typing.Sequence[typing.Sequence[int]] = [],
    ):
        super().__init__()
        self.joint_num = len(neighbors)
        if in_channels % self.joint_num != 0 or out_channels % self.joint_num != 0:
            log.error(
                "[SkeletonConvolution] Input and output channels should be a multiple of the number of joints."
            )
        self.in_channels_per_joint = in_channels // self.joint_num
        self.out_channels_per_joint = out_channels // self.joint_num
        if padding_mode == "zeros":
            padding_mode = "constant"

        self.expanded_neighbors = []
        self.expanded_neighbors_offset = []
        self.neighbors = neighbors

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = 1
        self.groups = 1
        self.padding = padding
        self.padding_mode = padding_mode
        self._padding_repeated_twice = (padding, padding)

        for neighbour in neighbors:
            expanded = []
            for k in neighbour:
                for i in range(self.in_channels_per_joint):
                    expanded.append(k * self.in_channels_per_joint + i)
            self.expanded_neighbors.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels, kernel_size)
        if bias:
            self.bias = torch.zeros(out_channels)
        else:
            self.register_parameter("bias", None)

        mask = torch.zeros_like(self.weight)
        for i, neighbour in enumerate(self.expanded_neighbors):
            mask[
                self.out_channels_per_joint * i : self.out_channels_per_joint * (i + 1),
                neighbour,
                ...,
            ] = 1
        # self.mask = torch.nn.parameter.Parameter(self.mask, requires_grad=False) #TODO: why param?
        self.register_buffer("mask", mask)
        self.reset_parameters()

    def reset_parameters(self):
        for i, neighbour in enumerate(self.expanded_neighbors):
            # NOTE: Use temporary variable to avoid assign to copy of slice, which might lead to un expected result
            tmp = torch.zeros_like(
                self.weight[
                    self.out_channels_per_joint
                    * i : self.out_channels_per_joint
                    * (i + 1),
                    neighbour,
                    ...,
                ]
            )
            torch.nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[
                self.out_channels_per_joint * i : self.out_channels_per_joint * (i + 1),
                neighbour,
                ...,
            ] = tmp
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                    self.weight[
                        self.out_channels_per_joint
                        * i : self.out_channels_per_joint
                        * (i + 1),
                        neighbour,
                        ...,
                    ]
                )
                bound = 1 / math.sqrt(fan_in)
                tmp = torch.zeros_like(
                    self.bias[
                        self.out_channels_per_joint
                        * i : self.out_channels_per_joint
                        * (i + 1)
                    ]
                )
                torch.nn.init.uniform_(tmp, -bound, bound)
                self.bias[
                    self.out_channels_per_joint
                    * i : self.out_channels_per_joint
                    * (i + 1)
                ] = tmp

        self.weight = torch.nn.parameter.Parameter(self.weight)
        if self.bias is not None:
            self.bias = torch.nn.parameter.Parameter(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight_masked = self.weight * self.mask
        return torch.nn.functional.conv1d(
            torch.nn.functional.pad(
                input, self._padding_repeated_twice, mode=self.padding_mode
            ),
            weight_masked,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )

    def __repr__(self):
        return (
            "SkeletonConv(in_channels_per_armature={}, out_channels_per_armature={}, kernel_size={}, "
            "joint_num={}, stride={}, padding={}, bias={})".format(
                self.in_channels_per_joint,
                self.out_channels_per_joint,
                self.kernel_size,
                self.joint_num,
                self.stride,
                self.padding,
                self.bias,
            )
        )
