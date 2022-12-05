import torch

# Paper: https://arxiv.org/pdf/2007.15240.pdf

class Diversity(torch.nn.Module):
    def __init__(self):
        super(Diversity, self).__init__()

    def forward(self,
        pred:       torch.Tensor,
        weights:    torch.Tensor=None,
    ) -> torch.Tensor:
        b = int(pred.shape[0] / 2)
        v1 = pred[:b]
        v2 = pred[b:2*b]
        diversity = torch.mean(torch.norm(v1 - v2, p=2, dim=-1, keepdim=False))
        if weights is not None:
            diversity = diversity * weights
        return diversity