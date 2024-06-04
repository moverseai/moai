import torch

__all__ = ["GradientPenalty"]


class GradientPenalty(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, blended_samples: torch.Tensor, blended_scores: torch.Tensor
    ) -> torch.Tensor:
        grad_outputs = torch.ones_like(blended_scores)
        gradients = torch.autograd.grad(
            outputs=blended_scores,
            inputs=blended_samples,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_norm = gradients.norm(p=2, dim=1)
        grad_penalty = torch.mean((grad_norm - 1) ** 2)
        # grad_penalty.backward(retain_graph=True)
        return grad_penalty
