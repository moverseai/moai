import typing

import toolz
import torch

from moai.utils.torch import get_submodule

__all__ = ["ModelParameterSelector"]


class ModelParameterSelector(
    typing.Callable[[torch.nn.Module], typing.List[torch.Tensor]]
):
    def __init__(
        self,
        modules: typing.Sequence[str] = [],
        monads: typing.Sequence[str] = [],
        parameters: typing.Sequence[str] = [],
        force_grad: bool = True,
    ):
        self.modules = modules or []
        self.monads = monads or []
        self.parameters = parameters or []
        self.force_grad = force_grad

    def __call__(
        self, moai_model: torch.nn.Module
    ) -> typing.Dict[str, typing.List[torch.Tensor]]:
        params = []
        for key in self.modules:
            m = get_submodule(moai_model.named_components, key)
            params.append(m.parameters())
        for key in self.monads:
            m = get_submodule(moai_model.named_flows, key)
            params.append(m.parameters())
        parameters = list(toolz.concat(params))
        for key in self.parameters:
            keys = key.split(".")
            m = get_submodule(moai_model.named_flows, ".".join(keys[:-1]))
            p = getattr(m, keys[-1])
            parameters.append(p)
        if self.force_grad:
            for p in parameters:
                p.requires_grad_(True)
        return {"params": parameters}
