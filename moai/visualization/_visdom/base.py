from collections.abc import Callable

import visdom
import typing
import torch

__all__ = ["Base"]

class Base(Callable):
    def __init__(self,
        name:           str="default",
        ip:             str="http://localhost",
        port:           int=8097, #NOTE: visdom's default port
    ):
        self.env_name, self.ip, self.port, self.viz = \
            name, ip, port, None

    @property
    def visualizer(self):
        if self.viz is not None:
            return self.viz

        self.viz = visdom.Visdom(
            server=self.ip,
            port=self.port,
            env=self.env_name,
            use_incoming_socket=False
        )
        return self.viz

    def __call__(self, 
        tensors:    typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None
    ) -> None:
        raise NotImplementedError()