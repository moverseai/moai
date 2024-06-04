import functools
import math
import typing

import torch

from moai.supervision.objectives.regression import L1

__all__ = ["Wing"]


class Wing(L1):
    r"""Implements the generalized robust Wing loss function, including its **adaptive** and **soft** variants.

    ??? note "Wing Loss"
        <p align="center"><img src="https://render.githubusercontent.com/render/math?math=%5Chuge%5Cbegin%7Bequation%7DWing%28x%29%3D%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7Dw%5Cln%28%201%20%2B%20|x|%2F%5Cepsilon%29%20%26%20%5Ctext%7Bif%7D%20|x|%20%3C%20w%20%5C%5C%20|x|%20-%20C%26%5Ctext%7Botherwise%7D%5Cend%7Barray%7D%5Cright.%5Cend%7Bequation%7D"/></p>

    ??? note "Adaptive Wing Loss"
        <p align="center"><img src="https://render.githubusercontent.com/render/math?math=%5Chuge%5Cbegin%7Bequation%7D%20%5Csmall%20AdaptiveWing%28y%2C%5C!%5Chat%7By%7D%29%5C!%20%3D%5C!%20%5Cbegin%7Bcases%7D%20%20%5Comega%5C!%20%5Cln%281%5C!%20%2B%5C!%20%20%5Cdisplaystyle%20|%5Cfrac%7By%5C!-%5C!%5Chat%7By%7D%5C!%7D%7B%5Cepsilon%7D|^%7B%5Calpha-y%7D%29%5C!%20%26%5C!%20%5Ctext%7Bif%20%7D%20|%28y%5C!-%5C!%5Chat%7By%7D%29|%5C!%20%3C%5C!%20%5Ctheta%20%20%20%5C%5C%20%20%20A|y-%5Chat%7By%7D%5C!|%20-%20C%20%26%20%5Ctext%7Botherwise%7D%20%5Cend%7Bcases%7D%20%5Cend%7Bequation%7D"/></p>

    ??? note "Soft Wing Loss"
        <p align="center"><img src="https://render.githubusercontent.com/render/math?math=%5Chuge%5Cbegin%7Bequation%7D%5Cmathrm%7BSoftWing%7D%28x%29%3D%5Cleft%5C%7B%20%5Cbegin%7Barray%7D%7Bll%7D%20%5Comega%20%5Cln%28%7B1%2B%5Cfrac%7B|x|%7D%7B%5Cepsilon%7D%7D%29%26%20%5Cmathrm%7Bif%7D%5C%20|x|%20%3C%20%5Comega%20%5C%5C%20|x|%20-%20C%20%20%26%5Cmathrm%7Botherwise%7D%20%5Cend%7Barray%7D%20%5Cright.%20%5Cend%7Bequation%7D"/></p>

    ??? cite "Papers"
        [![Paper](https://img.shields.io/static/v1?label=1711.06753&message=Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks&color=1c1ca2&logo=arxiv&style=plastic)](https://arxiv.org/pdf/1711.06753.pdf)

        [![Paper](https://img.shields.io/static/v1?label=1904.07399&message=Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression&color=1c1ca2&logo=arxiv&style=plastic)](https://arxiv.org/pdf/1904.07399.pdf)

        [![Paper](https://img.shields.io/static/v1?label=2006.11697&message=Fast and Accurate: Structure Coherence Component for Face Alignment&color=1c1ca2&logo=arxiv&style=plastic)](https://arxiv.org/pdf/2006.11697.pdf)

    ???+ important "Config Entry"
        === "Main Entry"
            ```yaml
            - model/supervision/losses/regression/robust: wing
            ```
        === "Parameters"
            ```yaml
            mode: standard # one of ['standard', 'adaptive', 'soft']
            omega: 0.05
            epsilon: 1.0
            theta: 0.05
            alpha: 2.1
            omega2: 0.02
            ```
        === "DML"
            ```yaml
            model:
              supervision:
                losses:
                  wing:
                    mode: standard # one of ['standard', 'adaptive', 'soft']
                    omega: 0.05
                    epsilon: 1.0
                    theta: 0.05
                    alpha: 2.1
                    omega2: 0.02
            ```
        === "Graph"
            ```yaml
            model:
              supervision:
                losses:
                  wing:
                    gt: [groundtruth_tensor_name] # the name of the mean tensor
                    pred: [prediction_tensor_name] # the name of the variance tensor
                    out: [wing_loss] # optional, will be 'wing' if omitted
            ```

    Arguments:
        mode (str, required):
            Wing variant selector parameter, can be one of [_standard_, _adaptive_, _soft_].
            See the respective papers for more details, each variant uses a different parameter subset.
            **Default value: standard.**
        omega (float, optional):
            The non-negative **ω** parameter that sets the range for the non-linear part, which is
            the turning point that switches between L1 and log loss.
            **Default value: 0.05.**
        epsilon (float):
            The curvature limiting factor **ε** for the robust Wing loss.
            **Default value: 1.0.**
        theta (float):
            The **θ** parameter that controls the switch between linear and nonlinear part for the Adaptive Wing Loss.
            **Default value: 0.05.**
        alpha (float):
            The parameter **α** used to adapt the shape of the loss function to make it smooth at point zero for the Adaptive Wing Loss.
            **Default value: 2.1.**
        omega2 (float):
            The non-negative **ω2** parameter that controls the threshold between medium and large errors for the Soft Wing Loss.
            **Default value: 0.02.**

    !!! important
        ω2 should not set to small value because it will cause gradient vanishing problem.

    !!! important
        Note that we should not set ω to a very small value because it makes the training of a network very unstable and causes the exploding gradient problem for very small errors.

    !!! important
        Parameter α has to be slightly larger than 2 to maintain the ideal properties required due to the normalization of the predicted variable in the range of [0, 1].

    ??? info "Repositories"
        [![Repo2](https://github-readme-stats.vercel.app/api/pin/?username=protossw512&repo=AdaptiveWingLoss&hide_border=true&title_color=1c1ca2&show_owner=true)](https://github.com/protossw512/AdaptiveWingLoss)
        [![Repo1](https://github-readme-stats.vercel.app/api/pin/?username=elliottzheng&repo=AdaptiveWingLoss&hide_border=true&title_color=1c1ca2&show_owner=true)](https://github.com/elliottzheng/AdaptiveWingLoss)
        [![Repo3](https://github-readme-stats.vercel.app/api/pin/?username=SeungyounShin&repo=Adaptive-Wing-Loss-for-Robust-Face-Alignment-via-Heatmap-Regression&hide_border=true&title_color=1c1ca2&show_owner=true)](https://github.com/SeungyounShin/Adaptive-Wing-Loss-for-Robust-Face-Alignment-via-Heatmap-Regression)

    """

    def __init__(
        self,
        mode: str = "standard",  # 'standard', 'adaptive', 'soft'
        omega: typing.Optional[float] = 0.05,
        epsilon: float = 1.0,
        theta: float = 0.05,  # variable θ as a threshold to switch between linear and nonlinear part
        alpha: float = 2.1,  # Note α has to be slightly larger than 2 to maintain the ideal properties
        omega2: float = 0.02,
    ):
        super(Wing, self).__init__()
        self.threshold = theta if mode == "adaptive" else omega
        Cw = omega - omega * math.log(1.0 + omega / epsilon)
        Ca = theta / epsilon
        Cs = omega - omega2 * math.log(1.0 + omega / epsilon)
        self.branch = (
            functools.partial(self._standard_branch, omega=omega, epsilon=epsilon, C=Cw)
            if mode == "standard"
            else (
                functools.partial(
                    self._adaptive_branch,
                    omega=omega,
                    epsilon=epsilon,
                    alpha=alpha,
                    theta=theta,
                    C=Ca,
                )
                if mode == "adaptive"
                else functools.partial(
                    self._soft_branch, omega=omega, epsilon=epsilon, omega2=omega2, C=Cs
                )
            )
        )

    @staticmethod
    def _adaptive_branch(
        L1: torch.Tensor,
        gt: torch.Tensor,
        omega: float,
        epsilon: float,
        alpha: float,
        theta: float,
        C: float,
    ) -> torch.Tensor:
        a_minus_y = alpha - gt
        A = (
            omega
            * (1.0 / (1.0 + C**a_minus_y))
            * a_minus_y
            * (C ** (a_minus_y - 1.0))
            / epsilon
        )
        C = theta * A - omega * torch.log(1.0 + C**a_minus_y)
        return omega * torch.log(1.0 + (L1 / epsilon) ** a_minus_y), A * L1 - C

    @staticmethod
    def _standard_branch(
        L1: torch.Tensor,
        gt: torch.Tensor,
        omega: float,
        epsilon: float,
        C: float,
    ) -> torch.Tensor:
        return omega * torch.log(1.0 + L1 / epsilon), L1 - C

    @staticmethod
    def _soft_branch(
        L1: torch.Tensor,
        gt: torch.Tensor,
        omega: float,
        epsilon: float,
        omega2: float,
        C: float,
    ) -> torch.Tensor:
        return L1, omega2 * torch.log(1 + L1 / epsilon) + C

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        weights: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        L1 = super(Wing, self).forward(pred=pred, gt=gt)
        branch_A, branch_B = self.branch(L1, gt)
        wing = torch.where(L1 < self.threshold, branch_A, branch_B)
        if weights is not None:
            wing = wing * weights
        if mask is not None:
            wing = wing[mask]
        return wing
