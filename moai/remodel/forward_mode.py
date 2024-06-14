import logging
import typing
from contextlib import ContextDecorator

import toolz
import torch

__all__ = ["ForwardMode"]

log = logging.getLogger(__name__)


class nograd_mode(torch.no_grad):
    def __init__(self, *args, **kwargs):
        super(nograd_mode, self).__init__()


class inference_mode(torch.inference_mode):
    def __init__(self, *args, **kwargs):
        super(inference_mode, self).__init__()


class eval_mode(ContextDecorator):
    def __init__(self, module: torch.nn.Module):
        self.module = module
        # self.module_train_state = False

    def __enter__(self):
        if self.module.training:
            # self.module_train_state = self.module.training
            self.module.eval()
            log.info(f"{type(self.module)} set to eval() mode.")

    def __exit__(
        self, exc_type: typing.Any, exc_value: typing.Any, traceback: typing.Any
    ) -> None:
        pass
        # if self.module_train_state:
        #     self.module.train(self.module_train_state)


class eval_nograd_mode(ContextDecorator):
    def __init__(self, module: torch.nn.Module):
        # super(eval_nograd_mode, self).__init__()
        self.module = module
        self.module_grad_state = False

    def __enter__(self):
        # self.module_train_state = self.module.training
        self.module_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        if self.module.training:
            self.module.eval()
            log.info(f"{type(self.module)} set to eval() and nograd() mode.")

    def __exit__(
        self, exc_type: typing.Any, exc_value: typing.Any, traceback: typing.Any
    ) -> None:
        torch.set_grad_enabled(self.module_grad_state)


class ForwardMode(typing.Callable[[torch.nn.Module], None]):

    __TYPE__ = {
        "nograd": nograd_mode,
        "no_grad": nograd_mode,
        "inference": inference_mode,
        "infer": inference_mode,
        "eval": eval_mode,
        "eval_nograd": eval_nograd_mode,
        "nograd_eval": eval_nograd_mode,
    }

    def __init__(
        self,
        modules: typing.Mapping[str, str] = None,
        monads: typing.Mapping[str, str] = None,
    ):
        self.modules = modules or {}
        self.monads = monads or {}

    def __call__(self, model: torch.nn.Module) -> None:
        for module, mode in self.modules.items():
            # split = module.split('.')
            # m = toolz.reduce(getattr, split, model)
            m = model.named_components.get_submodule(module)
            m.forward = ForwardMode.__TYPE__[mode](m)(m.forward)
            log.info(f"Remodeling the forward of {module} to {mode}")
        for monad, mode in self.monads.items():
            # split = module.split('.')
            # m = toolz.reduce(getattr, split, model)
            m = model.named_flows.get_submodule(monad)
            m.forward = ForwardMode.__TYPE__[mode](m)(m.forward)
            log.info(f"Remodeling the forward of monad [{monad}] to {mode}")
