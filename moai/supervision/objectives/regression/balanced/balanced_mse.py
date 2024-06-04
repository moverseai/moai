import logging

import torch
from torch.distributions import MultivariateNormal as MVN

log = logging.getLogger(__name__)

__all__ = [
    "ScalarMSE",
    "VectorMSE",
    "MVNMSE",
]


class ScalarMSE(torch.nn.Module):
    def __init__(
        self,
        sigma: float,
        mode: str = "auto",  # one of [auto, const]
    ):
        super(ScalarMSE, self).__init__()
        if mode == "auto":
            self.register_parameter(
                "sigma", torch.nn.Parameter(torch.scalar_tensor(sigma))
            )
        else:
            self.register_buffer(
                "sigma", torch.nn.Parameter(torch.scalar_tensor(sigma))
            )

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor = None,  # NOTE: w/o gt it serves as a prior
        weights: torch.Tensor = None,  # float tensor
        mask: torch.Tensor = None,  # byte tensor
    ) -> torch.Tensor:
        p = pred.flatten(start_dim=1).reshape(-1, 1)
        g = gt.flatten(start_dim=1).reshape(-1, 1)
        noise_var = self.sigma**2
        logits = -0.5 * (p - g.T).pow(2) / noise_var
        loss = torch.nn.functional.cross_entropy(
            logits, torch.arange(p.shape[0], device=p.device), reduction="none"
        )
        loss = loss * (2 * noise_var).detach()
        # if weights is not None:
        #     l2 = l2 * weights
        # if mask is not None:
        #     l2 = l2[mask]
        return loss


class VectorMSE(torch.nn.Module):
    def __init__(
        self,
        sigma: float,
        mode: str = "auto",  # one of [auto, const]
    ):
        super(VectorMSE, self).__init__()
        if mode == "auto":
            self.register_parameter(
                "sigma", torch.nn.Parameter(torch.scalar_tensor(sigma))
            )
        else:
            self.register_buffer(
                "sigma", torch.nn.Parameter(torch.scalar_tensor(sigma))
            )

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor = None,  # NOTE: w/o gt it serves as a prior
        weights: torch.Tensor = None,  # float tensor
        mask: torch.Tensor = None,  # byte tensor
    ) -> torch.Tensor:
        noise_var = self.sigma**2
        mylist = []
        inner = []
        for i in range(pred.shape[0]):
            inner.clear()
            for j in range(pred.shape[0]):
                v = torch.linalg.norm(gt[j] - pred[i], dim=-1).mean()
                inner.append(v)
            mylist.append(torch.stack(inner))
        norms = torch.stack(mylist)
        logits = -0.5 * (norms).pow(2) / noise_var
        loss = torch.nn.functional.cross_entropy(
            logits, torch.arange(pred.shape[0], device=pred.device), reduction="none"
        )
        loss = loss * (2 * noise_var).detach()
        if weights is not None:
            loss = loss * weights
        return loss


class MVNMSE(torch.nn.Module):
    def __init__(
        self,
        sigma: float = 0.0,
        dim: int = 1,
        mode: str = "auto",  # one of [auto, const]
    ):
        super(MVNMSE, self).__init__()
        if mode == "auto":
            self.register_parameter(
                "sigma",
                torch.nn.Parameter(
                    torch.scalar_tensor(sigma)
                    if dim == 1
                    else torch.tensor([sigma] * dim)
                ),
            )
        else:
            self.register_buffer(
                "sigma",
                torch.nn.Parameter(
                    torch.scalar_tensor(sigma)
                    if dim == 1
                    else torch.tensor([sigma] * dim)
                ),
            )

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor = None,  # NOTE: w/o gt it serves as a prior
        weights: torch.Tensor = None,  # float tensor
        mask: torch.Tensor = None,  # byte tensor
    ) -> torch.Tensor:
        # if pred.shape[0] == 1:
        #     pred, gt = pred[0], gt[0]
        if weights is not None:
            s = pred.shape
            ctrl = self.sigma.detach().view(*s[1:]).unsqueeze(0).expand_as(pred)
            pred = pred * weights + ctrl
            gt = gt * weights - ctrl
        if mask is not None:
            s = pred.shape
            ctrl = self.sigma.detach().view(*s[1:]).unsqueeze(0).expand_as(pred)
            pred[mask] = ctrl
            gt[mask] = -ctrl
        p = pred.flatten(start_dim=1)
        g = gt.flatten(start_dim=1)
        noise_var = self.sigma**2
        I = torch.eye(p.shape[-1]).to(gt.device)
        logits = MVN(p.unsqueeze(1), noise_var * I).log_prob(
            g.unsqueeze(0)
        )  # [B,1,N] / [N,N] / [1,B,N]  => [B, B]
        loss = torch.nn.functional.cross_entropy(
            logits, torch.arange(p.shape[0], device=p.device)
        )
        loss = loss * (2 * noise_var).detach()
        return loss
