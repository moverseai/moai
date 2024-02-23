# from moai.utils.arguments import assert_numeric

import torch
import logging
import typing

log = logging.getLogger(__name__)

__all__ = ["SO3Prior"]


def map_to_lie_algebra(v):
    """Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra
    Arg:
        v = vector in R^N, (..., 3) in our case
    Return:
        R = v converted to Lie Algebra element, (3,3) in our case"""

    # make sure this is a sample from R^3
    assert v.size()[-1] == 3

    R_x = v.new_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    R_y = v.new_tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    R_z = v.new_tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    R = (
        R_x * v[..., 0, None, None]
        + R_y * v[..., 1, None, None]
        + R_z * v[..., 2, None, None]
    )
    return R


def rodrigues(v):
    theta = v.norm(p=2, dim=-1, keepdim=True)
    # normalize K
    #TODO: what if theta is zero?
    K = map_to_lie_algebra(v / theta)
    Id = torch.eye(3, device=v.device, dtype=v.dtype)
    R = (
        Id
        + torch.sin(theta)[..., None] * K
        + (1.0 - torch.cos(theta))[..., None] * (K @ K)
    )
    return R


def s2s2_gram_schmidt(v1, v2):
    """Normalise 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix."""
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-5)
    e3 = torch.cross(e1, e2)
    return torch.stack([e1, e2, e3], 1)


class N0reparameterize(torch.nn.Module):
    """Reparametrize zero mean Gaussian Variable."""

    def __init__(self, input_dim, z_dim, fixed_sigma=None):
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.sigma_linear = torch.nn.Linear(input_dim, z_dim)
        self.return_means = False
        if fixed_sigma is not None:
            self.register_buffer("fixed_sigma", torch.tensor(fixed_sigma))
        else:
            self.fixed_sigma = None
        self.softplus = torch.nn.Softplus(1.0, 20.0)
        self.sigma = None
        self.z = None

    def forward(self, x, n=1):
        if self.fixed_sigma is not None:
            self.sigma = x.new_full((x.shape[0], self.z_dim), self.fixed_sigma)
        else:
            self.sigma = self.softplus(self.sigma_linear(x))
        self.z = self.nsample(n=n)
        return self.z, self.sigma

    def kl(self):
        return -0.5 * torch.sum(1 + 2 * self.sigma.log() - self.sigma**2, -1)

    def log_posterior(self):
        return self._log_posterior(self.z)

    def _log_posterior(self, z):
        return (
            torch.distributions.normal.Normal(torch.zeros_like(self.sigma), self.sigma)
            .log_prob(z)
            .sum(-1)
        )

    def log_prior(self):
        return (
            torch.distributions.normal.Normal(
                torch.zeros_like(self.sigma), torch.ones_like(self.sigma)
            )
            .log_prob(self.z)
            .sum(-1)
        )

    def nsample(self, n=1):
        if self.return_means:
            return torch.zeros_like(self.sigma).expand(n, -1, -1)
        eps = torch.distributions.normal.Normal(
            torch.zeros_like(self.sigma), torch.ones_like(self.sigma)
        ).sample((n,))
        return eps * self.sigma


class S2S2Mean(torch.nn.Module):
    """Module to map R^6 -> SO(3) with S2S2 method."""

    def __init__(self, input_dims, output_dims=6):
        super().__init__()
        assert output_dims == 6
        self.map = torch.nn.Linear(input_dims, output_dims)

        # Start with big outputs
        self.map.weight.data.uniform_(-10, 10)
        self.map.bias.data.uniform_(-10, 10)

    def forward(self, x):
        v = self.map(x).double().view(-1, 2, 3)
        v1, v2 = v[:, 0], v[:, 1]
        return s2s2_gram_schmidt(v1, v2).float()


class SO3Prior(torch.nn.Module):
    """Reparametrize SO(3) latent variable.

    It uses an inner zero mean Gaussian reparametrization module, which
    exp-maps to an identity centered random SO(3) variable. The mean_module
    deterministically outputs a mean.
    """

    def __init__(
        self,
        # input_dim: int,
        # output_dim: int,
        k=10,
        n_samples=1,
    ) -> None:
        super(SO3Prior, self).__init__()

        # self.mean_module = S2S2Mean(input_dim)
        # self.reparameterize = N0reparameterize(input_dim, output_dim)
        # self.sigma = self.reparameterize.sigma
        # assert self.reparameterize.z_dim == 3
        self.k = k
        self.return_means = False
        self.n = n_samples

    def forward(self, mu_lie: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # num of samples, batch, 3
        z = self.nsample(mu_lie=mu_lie, v=v, n=self.n)  # num of samples, batch, 3, 3
        # convert to num of samples*batch, 3, 3
        z = z.view(-1, *z.shape[2:])
        n = z.shape[0]
        return z.view(n, -1)  # num of samples*batch, 9
        # return z
        # why squeeze?
        # missing the n_samples dimension
        # return self.v, sigma, self.z #self.z.reshape(self.z.shape[0],-1) # why reshape?

    def nsample(self, mu_lie, v, n=1):
        # if self.return_means:
        #     return self.mu_lie.expand(n, *[-1]*len(self.mu_lie.shape))
        v_lie = rodrigues(v)
        return mu_lie @ v_lie
