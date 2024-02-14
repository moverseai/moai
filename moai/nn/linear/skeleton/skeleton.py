import math
import torch
import typing

class SkeletonLinear(torch.nn.Module):
    def __init__(self, 
        in_channels:        int, 
        out_channels:       int, 
        neighbour_list:     typing.Sequence[typing.Sequence[int]], 
    ):
        super().__init__()
        self.neighbour_list = neighbour_list
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_per_joint = in_channels // len(neighbour_list)
        self.out_channels_per_joint = out_channels // len(neighbour_list)
        self.expanded_neighbour_list = []

        for neighbour in neighbour_list:
            expanded = []
            for k in neighbour:
                for i in range(self.in_channels_per_joint):
                    expanded.append(k * self.in_channels_per_joint + i)
            self.expanded_neighbour_list.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels)
        self.mask = torch.zeros(out_channels, in_channels)
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            tmp = torch.zeros_like(
                self.weight[i*self.out_channels_per_joint: (i + 1)*self.out_channels_per_joint, neighbour]
            )
            self.mask[i*self.out_channels_per_joint: (i + 1)*self.out_channels_per_joint, neighbour] = 1
            torch.nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[i*self.out_channels_per_joint: (i + 1)*self.out_channels_per_joint, neighbour] = tmp

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self.weight = torch.nn.parameter.Parameter(self.weight) 
        self.mask = torch.nn.parameter.Parameter(self.mask, requires_grad=False) #NOTE: buffer?

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.reshape(input.shape[0], -1)#NOTE: flatten?
        weight_masked = self.weight * self.mask #NOTE: mask can be a buffer
        return torch.nn.functional.linear(input, weight_masked, self.bias)