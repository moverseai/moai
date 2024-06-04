import collections
import inspect
import itertools
import logging
import typing
from operator import add

import hydra.utils as hyu
import numpy as np
import omegaconf.omegaconf
import toolz
import torch

from moai.utils.arguments import ensure_numeric_list, ensure_string_list

log = logging.getLogger(__name__)

__all__ = ["Homoscedastic"]

from moai.monads.execution.cascade import _create_accessor

__REDUCTIONS__ = {
    "sum": torch.sum,
    "mean": torch.mean,
    "iou": lambda t: torch.mean(1.0 - (t / t[0].numel())),
}


# NOTE: https://arxiv.org/pdf/1705.07115.pdf
# NOTE: https://paperswithcode.com/paper/multi-task-learning-using-uncertainty-to
class Homoscedastic(torch.nn.ModuleDict):
    def __init__(
        self,
        losses: omegaconf.DictConfig,
        tasks: omegaconf.DictConfig,
        **kwargs: typing.Mapping[str, typing.Any],
    ):
        super(Homoscedastic, self).__init__()
        self.execs, self.weights, self.reductions = [], [], []
        self.tasks, self.scales = collections.defaultdict(list), {}
        for task, details in tasks.items():
            sigma = float(
                details.sigma
                if isinstance(details.sigma, (float, int))
                else np.random.uniform(*details.sigma)
            )
            self.register_parameter(
                task, torch.nn.parameter.Parameter(torch.scalar_tensor(sigma**2))
            )
            self.scales[task] = details.get("scale", 1.0)
            self.tasks[task] += (
                [details.losses] if isinstance(details.losses, str) else details.losses
            )
        if not len(losses):
            log.warning(
                "A weighted combination of losses is being used for supervising the model, but no losses have been assigned."
            )
        if not len(tasks):
            log.warning(
                "A multi-task supervision scheme is being used but no task-specific details have been assigned."
            )
        loop = ((key, params) for key, params in kwargs.items() if key in losses)
        errors = [k for k in kwargs if k not in losses]
        if errors:
            log.error(
                f"Some losses [{''.join(errors)}] were not found in the configuration and will be ignored!"
            )
        self.keyz = []
        for k, p in loop:
            self.add_module(k, hyu.instantiate(getattr(losses, k)))
            # last_module = toolz.last(self.modules()) # moduledict is ordered
            last_module = self[k]
            sig = inspect.signature(last_module.forward)
            p = toolz.valmap(ensure_string_list, p)
            if "out" not in p:
                length = len(ensure_string_list(next(iter(p.values()))))
                p["out"] = [k] if length == 1 else [f"{k}_{i}" for i in range(length)]
            if "weight" in p:
                wgts = iter(ensure_numeric_list(p["weight"]))
            else:
                log.warning(
                    f"{k} loss has no assigned weights, automatically reverting to a weight of one (1.0)."
                )
                wgts = itertools.cycle([1.0 / len(p["out"])])
            if "reduction" in p:
                reduction = iter(p["reduction"])
            else:
                log.warning(
                    f"{k} loss has no assigned reduction, automatically reverting to mean reduction."
                )
                reduction = itertools.cycle(["mean"])
            # TODO: there is a bug if you pass in keys that are not bracketed ([]), i.e. as a list, even for a single arg
            for keys in zip(
                *list(
                    p[prop]
                    for prop in itertools.chain(sig.parameters, ["out"])
                    if p.get(prop) is not None
                )
            ):
                accessors = [
                    _create_accessor(k if isinstance(k, str) else toolz.get(0, k, None))
                    for k in keys[:-1]
                ]
                self.execs.append(
                    lambda tensor_dict, acc=accessors, k=keys, p=sig.parameters.keys(), f=last_module: tensor_dict.update(
                        {
                            k[-1]: f(
                                **dict(
                                    zip(
                                        p,
                                        # list(tensor_dict[i] for i in k[:-1])
                                        # list(tensor_dict.get(i, None)
                                        list(
                                            a(tensor_dict)
                                            for a, i in zip(acc, k[:-1])
                                            if i is not None or None
                                        ),
                                    )
                                )
                            )
                        }
                    )
                )
                self.keyz.append(keys[-1])
                self.weights.append(
                    next(wgts)
                )  # TODO: error if no weight has been set? or implicit 1.0 ?
                self.reductions.append(next(reduction))

    def forward(self, tensors: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(
            toolz.take(
                1, filter(lambda t: isinstance(t, torch.Tensor), tensors.values())
            )
        ).device
        error = torch.tensor(0.0, dtype=torch.float32, device=device)
        per_error_map = {}
        for exe, w, k, r in zip(self.execs, self.weights, self.keyz, self.reductions):
            exe(tensors)
            e = w * __REDUCTIONS__[r](tensors[k])
            per_error_map[k] = e
        for n, p in self.named_parameters():
            weight = 1.0 / (2.0 * p)
            loss = sum((per_error_map.get(l, 0.0) for l in self.tasks[n]), 0.0)
            scale = self.scales[n]
            is_float_zero = isinstance(loss, (float)) and loss == 0.0
            is_unoptimized = isinstance(loss, torch.Tensor) and not loss.requires_grad
            if not is_float_zero and not is_unoptimized:
                error += (weight * (scale * loss)) + torch.log(p)
                per_error_map[f"{n}_uncertainty"] = p
        return error, per_error_map
