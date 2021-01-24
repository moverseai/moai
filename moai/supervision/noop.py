import torch
import typing

class NoOp(torch.nn.ModuleDict):
    def __init__(self):
        super(NoOp, self).__init__()

    def forward(self, 
        tensors: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        return torch.scalar_tensor(0.0), { }