from moai.monads.utils import dim_list
from moai.utils.arguments import assert_numeric

import torch
import functools
import logging

log = logging.getLogger(__name__)

#NOTE: beta: from https://openreview.net/forum?id=Sy2fzU9gl
#NOTE: capacity: from https://arxiv.org/pdf/1804.03599.pdf
#NOTE: robust: from https://arxiv.org/pdf/2007.13886.pdf

__all__ = ["StandardNormalKL"]

class StandardNormalKL(torch.nn.Module):
    r""" Implements the generalized Kullback-Leibler (KL) divergence function assuming the standard normal distribution as prior, including its **beta** (β-VAE), **capacity** (disentangled β-VAE), and **robust** (Charbonnier) variants. The method receives the mean (μ) and the variance (σ) of the input distribution as input and returns the KL divergence.

    ??? note "Standard KL Loss"
        <p align="center"><img src="https://render.githubusercontent.com/render/math?math=%5Chuge%5Cbegin%7Bequation%7D%5Cmathrm%7BStandardKL%7D%28%5Cmu%2C%5Csigma%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Cdisplaystyle%5Csum_%7Bi%7D%20%281%2B%5Clog%28%5Csigma_%7Bi%7D^%7B2%7D%29%20-%5Cmu_%7Bi%7D^%7B2%7D-%5Csigma_%7Bi%7D^%7B2%7D%29%5Cend%7Bequation%7D"/></p>

    ??? note "Beta-Standard KL Loss"
        <p align="center"><img src="https://render.githubusercontent.com/render/math?math=%5Chuge%5Cbegin%7Bequation%7D%5Cbeta%5Cmathrm%7B-StandardKL%7D%28%5Cmu%2C%5Csigma%29%20%3D%20%5Cbeta%20%5C%2C%20%5C%2C%20%5Cfrac%7B1%7D%7B2%7D%20%5Cdisplaystyle%5Csum_%7Bi%7D%20%281%2B%5Clog%28%5Csigma_%7Bi%7D^%7B2%7D%29%20-%5Cmu_%7Bi%7D^%7B2%7D-%5Csigma_%7Bi%7D^%7B2%7D%29%5Cend%7Bequation%7D"/></p>

    ??? note "Capacity Standard KL Loss"
        <p align="center"><img src="https://render.githubusercontent.com/render/math?math=%5Chuge%5Cbegin%7Bequation%7D%5Cmathrm%7BCapacityStandardKL%7D%28%5Cmu%2C%5Csigma%29%20%3D%20%5Cbeta%20%5C%2C%20%5C%2C%20%5C%2C%20|%5C%2C%5Cfrac%7B1%7D%7B2%7D%20%5Cdisplaystyle%5Csum_%7Bi%7D%20%281%2B%5Clog%28%5Csigma_%7Bi%7D^%7B2%7D%29%20-%5Cmu_%7Bi%7D^%7B2%7D-%5Csigma_%7Bi%7D^%7B2%7D%29%5C%2C-%5C%2CC%5C%2C|%5Cend%7Bequation%7D"/></p>

    ??? note "Robust Standard KL Loss"
        <p align="center"><img src="https://render.githubusercontent.com/render/math?math=%5Chuge%5Cbegin%7Bequation%7D%5Cmathrm%7BRobustStandardKL%7D%28%5Cmu%2C%5Csigma%29%20%3D%20%5Csqrt%7B1%20%2B%20%5Cbig%28%5Cfrac%7B1%7D%7B2%7D%20%5Cdisplaystyle%5Csum_%7Bi%7D%20%281%2B%5Clog%28%5Csigma_%7Bi%7D^%7B2%7D%29%20-%5Cmu_%7Bi%7D^%7B2%7D-%5Csigma_%7Bi%7D^%7B2%7D%29%5Cbig%29^2%7D%20-%201%5Cend%7Bequation%7D"/></p>

    ??? cite "Papers"
        [![Paper](https://img.shields.io/static/v1?label=OpenReview&message=beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework&color=1c1ca2&)](https://openreview.net/forum?id=Sy2fzU9gl)

        [![Paper](https://img.shields.io/static/v1?label=1904.07399&message=Understanding disentangling in β-VAE&color=1c1ca2&logo=arxiv)](https://arxiv.org/pdf/1804.03599.pdf)
        
        [![Paper](https://img.shields.io/static/v1?label=2006.11697&message=Perpetual Motion: Generating Unbounded Human Motion&color=1c1ca2&logo=arxiv)](https://arxiv.org/pdf/2007.13886.pdf)

    ???+ example "Configuration"
        === "Main Entry"
            ```yaml
            - model/supervision/loss/classification/distribution:  std_kl
            ```
        === "Parameters"
            ```yaml
            mode: 'standard'
            is_var_log: true
            beta: 1.0
            C_max: 0
            C_max_iter: 0      
            ```
        === "DML"
            ```yaml
            model:
              supervision:
                losses:
                  std_kl:
                    mode: capacity
                    is_var_log: true
                    beta: 100.0
                    C_max: 25
                    C_max_iter: 1e4
            ```
        === "Graph"
            ```yaml
            model:
              supervision:
                losses:
                  std_kl:
                    mu: [input_distribution_mean_tensor_name] # the name of the mean tensor
                    var: [input_distribution_variance_tensor_name] # the name of the variance tensor
                    out: [std_kl_loss] # optional, will be 'std_kl' if omitted
            ```
    
    Arguments:
        mode (str, required):
            Kullback-Leibler divergence between the input distribution and the standard normal one. Selector parameter can be one of [standard, beta, capacity, robust]. 
            See the respective papers for more details, each variant uses a different parameter subset.
            **Default value: standard.**
        is_var_log: (boolean, required):
            Incidates if the input is already in logarithmic scale (True) or not (False).
            **Default value: True**
        beta (float): 
            Represents the regularisation coefficient **β** that constrains the capacity of the latent space of a VAE model
            **Default value: 1.0**
        C_max (int):
            The maximum value of the factor capacity. Used in combination with 'capacity' mode for the disentangled VAE model
            **Default value: 0**
        C_max_iter (int):
            The iteration where the capacity value stops to increase. Used in combination with 'capacity' mode for the disentangled VAE model
            **Default value: 0**

    !!! important
        β should be set to 1.0 when using 'standard' and 'robust' modes.

    !!! important
        C_max and C_max_iter should be used only in the context of 'capacity' mode with β >> 1.0

    !!! important
        The '_robust_' variant uses the [Charbonier penalty](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.469.129&rep=rep1&type=pdf) to prevent posterior collapse.

    
    """
    def  __init__(self,
        mode:           str = 'standard', # ['standard', 'beta', 'capacity', 'robust']
        is_var_log:     bool=True,
        beta:           float=1.0, #hyperparameter for the betaVAE case
        C_max:          int=0, #total capacity hyperparameter for the disentangled betaVAE case
        C_max_iter:     int=1e5, #hyperparameter for the disentangled betaVAE (capacity) case
    ):
        super(StandardNormalKL, self).__init__()
        assert_numeric(log, 'C_max', C_max, 0.0, None)
        assert_numeric(log, 'C_max_iter', C_max_iter, 0.0, None)
        self.is_var_log = is_var_log
        self.beta = beta
        self.C_max_iter = C_max_iter
        self.num_iter = 0
        self.register_buffer("C_max", torch.Tensor([C_max]))
        self.loss = self.loss_disentangled_beta if mode == 'capacity'\
            else (self.loss_robust_charbonnier if mode == 'robust'\
                else self.loss_beta
            )

    @staticmethod
    def loss_beta(
        mu:  torch.Tensor,
        var:        torch.Tensor,
        C_max:      torch.Tensor,
        C_max_iter: float,
        beta:       float,
        num_iter:   int
    ) -> torch.Tensor:
        normal_kl = 1.0 + var - mu.pow(2) - var.exp()
        return -0.5 * torch.sum(normal_kl, dim=dim_list(var)) * beta

    @staticmethod
    def loss_disentangled_beta(
        mu:  torch.Tensor,
        var:        torch.Tensor,
        C_max:      torch.Tensor,
        C_max_iter: float,
        beta:       float,
        num_iter:   int
    ) -> torch.Tensor:
        normal_kl = 1.0 + var - mu.pow(2) - var.exp()
        C = torch.clamp(C_max/C_max_iter * num_iter, 0, C_max.data[0])
        return torch.abs(-0.5 * torch.sum(normal_kl, dim=dim_list(var)) - C) * beta

    @staticmethod
    def loss_robust_charbonnier(
        mu:  torch.Tensor,
        var:        torch.Tensor,
        C_max:      torch.Tensor,
        C_max_iter: float,
        beta:       float,
        num_iter:   int
    ) -> torch.Tensor:
        normal_kl = 1.0 + var - mu.pow(2) - var.exp()
        s =  -0.5 * normal_kl * beta
        charbonnier_s  = torch.sqrt(1 + s.pow(2)) - 1
        return torch.sum(charbonnier_s, dim=dim_list(var))

    def forward(self,
        var: torch.Tensor,
        mu: torch.Tensor
    ) -> torch.Tensor:
        self.num_iter += 1
        v = var if self.is_var_log else var.log()
        return self.loss(mu, v, self.C_max, self.C_max_iter, self.beta, self.num_iter)
