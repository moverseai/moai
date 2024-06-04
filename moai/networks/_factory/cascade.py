import inspect
import logging
import typing
from collections import OrderedDict

import hydra.utils as hyu
import omegaconf.omegaconf
import torch

import moai.networks.lightning as minet
import moai.utils.parsing.rtp as mirtp

__all__ = ["Cascade"]

log = logging.getLogger(__name__)


def _create_processing_block(
    cfg: omegaconf.DictConfig, attribute: str, monads: omegaconf.DictConfig
):
    if not cfg and attribute in cfg:
        log.warning(f"Empty processing block ({attribute}) in feedforward model.")
    return (
        hyu.instantiate(getattr(cfg, attribute), monads)
        if cfg and attribute in cfg
        else torch.nn.Identity()
    )


class Cascade(minet.FeedForward):
    def __init__(
        self,
        configuration: omegaconf.DictConfig,
        modules: omegaconf.DictConfig,
        data: omegaconf.DictConfig = None,
        parameters: omegaconf.DictConfig = None,
        feedforward: omegaconf.DictConfig = None,
        monads: omegaconf.DictConfig = None,
        supervision: omegaconf.DictConfig = None,
        validation: omegaconf.DictConfig = None,
        visualization: omegaconf.DictConfig = None,
        export: omegaconf.DictConfig = None,
    ):
        super(Cascade, self).__init__(
            data=data,
            parameters=parameters,
            feedforward=feedforward,
            monads=monads,
            supervision=supervision,
            validation=validation,
            visualization=visualization,
            export=export,
        )

        self.mods = torch.nn.ModuleDict(
            OrderedDict(
                (k, hyu.instantiate(d))
                for k, d in sorted(modules.items(), key=lambda kvp: kvp[0])
            )
        )

        self.mod_fwds = OrderedDict()
        for name, module in self.mods.items():
            params = inspect.signature(module.forward).parameters
            mod_in = list(
                zip(
                    *[
                        mirtp.force_list(configuration[name][prop])
                        for prop in params
                        if configuration[name][prop] is not None
                    ]
                )
            )
            # NOTE: mod_in receives them ordered (list), and so args cannot be ignored (nulled) when not last in the list
            # TODO: fix it via mapping
            mod_out = mirtp.split_as(
                mirtp.resolve_io_config(configuration[name]["out"]), mod_in
            )

            mod_res_fill = [mirtp.get_result_fillers(module, out) for out in mod_out]
            get_mod_filler = iter(mod_res_fill)

            self.mod_fwds[name] = []
            for keys in mod_in:  # TODO: check if feedforward to ignore filler
                self.mod_fwds[name].append(
                    lambda td, tk=keys, args=params.keys(), mod=module, filler=next(
                        get_mod_filler
                    ): filler(
                        td,
                        mod(
                            **dict(
                                zip(
                                    args,
                                    list(
                                        (
                                            td[k]
                                            if type(k) is str
                                            else list(td[j] for j in k)
                                        )
                                        for k in tk
                                    ),
                                )
                            )
                        ),
                    )
                )
        self.pre_module = torch.nn.ModuleList(
            _create_processing_block(feedforward, f"pre_module{i}", monads)
            for i in range(1, len(self.mods) + 1)
        )

    def forward(
        self, td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for (k, v), pre in zip(
            self.mod_fwds.items(),
            self.pre_module,
        ):
            for d in v:
                td = pre(td)
                d(td)
        return td
