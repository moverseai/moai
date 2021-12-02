import torch
import typing
import logging
import toolz
from contextlib import ContextDecorator

__all__ = ['ModuleMode']

log = logging.getLogger(__name__)

#TODO: check inference_mode from pytorch 1.10

class eval_mode(ContextDecorator):
    def __init__(self, module: torch.nn.Module):
        self.module = module
        # self.module_train_state = False

    def __enter__(self):
        if self.module.training:
            # self.module_train_state = self.module.training
            self.module.eval()
            log.info(f'{type(self.module)} set to eval() mode.')

    def __exit__(self,
        exc_type: typing.Any,
        exc_value: typing.Any,
        traceback: typing.Any
    ) -> None:
        pass
        # if self.module_train_state:
        #     self.module.train(self.module_train_state)

class eval_nograd_mode(ContextDecorator):
    def __init__(self, module: torch.nn.Module):
        # super(eval_nograd_mode, self).__init__()
        self.module = module

    def __enter__(self):
        # self.module_train_state = self.module.training
        self.module_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        if self.module.training:
            self.module.eval()
            log.info(f'{type(self.module)} set to eval() mode.')
        
    def __exit__(self,
        exc_type: typing.Any,
        exc_value: typing.Any,
        traceback: typing.Any
    ) -> None:
        torch.set_grad_enabled(self.prev)

class ModuleMode(typing.Callable[[torch.nn.Module], None]):
    
    __TYPE__ = {
        'nograd':              torch.no_grad,
        'eval':                eval_mode,
        'eval_nograd':         eval_nograd_mode,
        'nograd_eval':         eval_nograd_mode,
    }

    def __init__(self,
        modules:    typing.Mapping[str, str],
    ):
        self.modules = modules

    def __call__(self, model: torch.nn.Module) -> None:
        for module, mode in self.modules.items():
            split = module.split('.')
            m = toolz.reduce(getattr, split, model)
            m.forward = ModuleMode.__TYPE__[mode](m)(m.forward)
            log.info(f"Transforming module {module}'s forward to {mode}")
