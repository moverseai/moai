import torch

class GetUV(torch.nn.Module):
    def __init__(self,
    ):
        super(GetUV, self).__init__()

    def forward(self,
        tensor:         torch.Tensor,
    ) -> torch.Tensor:
        return tensor[...,:2]