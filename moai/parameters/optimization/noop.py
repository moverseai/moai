import torch
import typing

__all__ = ['NoOp']

class Null(torch.optim.Optimizer):
  def __init__(self,
    parameters: typing.Iterator[torch.nn.Parameter],
  ):
    super(Null, self).__init__(parameters, {"lr": 0.0, "eps": 1e-8})

  def step(self, closure=None):
    if closure is not None:
      closure()
    return None

class NoOp(object): 
  def __init__(self, 
      parameters: typing.Iterator[torch.nn.Parameter],
  ):
    self.optimizers = [Null(parameters)]

  def step(self, closure=None):
    return None