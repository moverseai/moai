import torch

class Passthrough(torch.nn.Module):
    def __init__(self):
        super(Passthrough, self).__init__()

    def forward(self,
        feat:       torch.Tensor,
        weights:    torch.Tensor=None,
        mask:       torch.Tensor=None,
    ) -> torch.Tensor:
       return feat