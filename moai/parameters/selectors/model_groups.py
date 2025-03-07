import typing

import toolz
import torch

from moai.parameters.selectors.model import ModelParameterSelector

__all__ = ["ModelGroupParameterSelector"]


class ModelGroupParameterSelector(
    typing.Callable[[torch.nn.Module], typing.List[torch.Tensor]]
):
    def __init__(
        self,
        groups: typing.Mapping[
            str, typing.Any
        ],  # dict of model selector params & extra optim params
    ):
        self.groups = groups

    def __call__(
        self, moai_model: torch.nn.Module
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        all_params = []
        for name, options in self.groups.items():
            selector = ModelParameterSelector(
                modules=options.get("modules", None),
                monads=options.get("monads", None),
                parameters=options.get("parameters", None),
                force_grad=options.get("force_grad", True),
            )
            d = selector(moai_model)
            d["name"] = name
            d.update(
                **toolz.dissoc(options, "modules", "monads", "parameters", "force_grad")
            )
            all_params.append(d)
        return all_params
