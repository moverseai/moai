import typing

import numpy as np
import torch

__all__ = ["UniformSimulatedAnnealing", "GaussianSimulatedAnnealing"]

# NOTE: modified from https://github.com/jramapuram/SimulatedAnnealing


class UniformSampler(object):
    def __init__(
        self, minval: float, maxval: float, dtype: str = "float", cuda: bool = False
    ):
        self.minval = minval
        self.maxval = maxval
        self.cuda = cuda
        self.dtype_str = dtype
        dtypes = {
            "float": torch.cuda.FloatTensor if cuda else torch.FloatTensor,
            "int": torch.cuda.IntTensor if cuda else torch.IntTensor,
            "long": torch.cuda.LongTensor if cuda else torch.LongTensor,
        }
        self.dtype = dtypes[dtype]

    def sample(self, size):
        if self.dtype_str == "float":
            return self.dtype(*size).uniform_(self.minval, self.maxval)
        elif self.dtype_str == "int" or self.dtype_str == "long":
            return self.dtype(*size).random_(self.minval, self.maxval + 1)
        else:
            raise Exception("unknown dtype")


class GaussianSampler(object):
    def __init__(
        self, mu: float, sigma: float, dtype: str = "float", cuda: bool = False
    ):
        self.sigma = sigma
        self.mu = mu
        self.cuda = cuda
        self.dtype_str = dtype
        dtypes = {
            "float": torch.cuda.FloatTensor if cuda else torch.FloatTensor,
            "int": torch.cuda.IntTensor if cuda else torch.IntTensor,
            "long": torch.cuda.LongTensor if cuda else torch.LongTensor,
        }
        self.dtype = dtypes[dtype]

    def sample(self, size):
        """pytorch doesnt support int or long normal distrs
        so we will resolve to casting"""
        rand_float = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        rand_block = rand_float(*size).normal_(self.mu, self.sigma)

        if self.dtype_str == "int" or self.dtype_str == "long":
            rand_block = rand_block.type(self.dtype)

        return rand_block


class SimulatedAnnealing(torch.optim.Optimizer):
    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        sampler: typing.Union[UniformSampler, GaussianSampler],
        tau0: float = 1.0,
        anneal_rate: float = 0.0003,
        min_temp: float = 1e-5,
        anneal_every: int = 100000,
        hard: bool = False,
        hard_rate: float = 0.9,
    ):
        defaults = dict(
            sampler=sampler,
            tau0=tau0,
            tau=tau0,
            anneal_rate=anneal_rate,
            min_temp=min_temp,
            anneal_every=anneal_every,
            hard=hard,
            hard_rate=hard_rate,
            iteration=0,
        )
        super(SimulatedAnnealing, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise Exception("loss closure is required to do SA")

        loss = closure()

        for group in self.param_groups:
            # the sampler samples randomness
            # that is used in optimizations
            sampler = group["sampler"]

            # clone all of the params to keep in case we need to swap back
            cloned_params = [p.clone() for p in group["params"]]

            for p in group["params"]:
                # anneal tau if it matches the requirements
                if (
                    group["iteration"] > 0
                    and group["iteration"] % group["anneal_every"] == 0
                ):
                    if not group["hard"]:
                        # smoother annealing: consider using this over hard annealing
                        rate = -group["anneal_rate"] * group["iteration"]
                        group["tau"] = np.maximum(
                            group["tau0"] * np.exp(rate), group["min_temp"]
                        )
                    else:
                        # hard annealing
                        group["tau"] = np.maximum(
                            group["hard_rate"] * group["tau"], group["min_temp"]
                        )

                random_perturbation = group["sampler"].sample(p.data.size())
                p.data = p.data / torch.norm(p.data)
                p.data.add_(random_perturbation)
                group["iteration"] += 1

            # re-evaluate the loss function with the perturbed params
            # if we didn't accept the new params swap back and return
            loss_perturbed = closure()
            final_loss, is_swapped = self.anneal(loss, loss_perturbed, group["tau"])
            if is_swapped:
                for p, pbkp in zip(group["params"], cloned_params):
                    p.data = pbkp.data

            return final_loss

    def anneal(self, loss, loss_perturbed, tau):
        """returns loss, is_new_loss"""

        def acceptance_prob(old, new, temp):
            return torch.exp((old - new) / temp)

        if loss_perturbed.data[0] < loss.data[0]:
            return loss_perturbed, True
        else:
            # evaluate the metropolis criterion
            ap = acceptance_prob(loss, loss_perturbed, tau)
            log.info(
                f"old = {loss.data[0]} | pert = {loss_perturbed.data[0]} | ap = {ap.data[0]} | tau = {tau}"
            )
            if ap.data[0] > np.random.rand():
                return loss_perturbed, True

            # return the original loss if above fails
            # or if the temp is now annealed
            return loss, False


class UniformSimulatedAnnealing(SimulatedAnnealing):
    """Implements Simulated Annealing optimization with uniform sampling.

    - **Paper**: [Optimization by Simulated Annealing](https://www.sciencedirect.com/science/article/pii/B9780080515816500593)
    - **Implementation**: [GitHub @ jramapuram](https://github.com/jramapuram/SimulatedAnnealing)

    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        minval: float,
        maxval: float,
        tau0: float = 1.0,
        anneal_rate: float = 0.0003,
        min_temp: float = 1e-5,
        anneal_every: int = 100000,
        hard: bool = False,
        hard_rate: float = 0.9,
        dtype: str = "float",
        cuda: bool = False,
    ):
        super(UniformSimulatedAnnealing, self).__init__(
            params,
            sampler=UniformSampler(
                minval=minval,
                maxval=maxval,
                dtype=dtype,
                cuda=cuda,
            ),
            tau0=tau0,
            anneal_rate=anneal_rate,
            anneal_every=anneal_every,
            min_temp=min_temp,
            hard=hard,
            hard_rate=hard_rate,
        )


class GaussianSimulatedAnnealing(SimulatedAnnealing):
    """Implements Simulated Annealing optimization with Gaussian sampling.

    - **Paper**: [Optimization by Simulated Annealing](https://www.sciencedirect.com/science/article/pii/B9780080515816500593)
    - **Implementation**: [GitHub @ jramapuram](https://github.com/jramapuram/SimulatedAnnealing)

    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        mu: float,
        sigma: float,
        tau0: float = 1.0,
        anneal_rate: float = 0.0003,
        min_temp: float = 1e-5,
        anneal_every: int = 100000,
        hard: bool = False,
        hard_rate: float = 0.9,
        dtype: str = "float",
        cuda: bool = False,
    ):
        super(GaussianSimulatedAnnealing, self).__init__(
            params,
            sampler=GaussianSampler(
                mu=mu,
                sigma=sigma,
                dtype=dtype,
                cuda=cuda,
            ),
            tau0=tau0,
            anneal_rate=anneal_rate,
            anneal_every=anneal_every,
            min_temp=min_temp,
            hard=hard,
            hard_rate=hard_rate,
        )
