import abc
import torch
import typing
import numpy as np

__all__ = ["MoaiMetric"]

#NOTE: should not use hooks /w metrics, they will not be called
class MoaiMetric(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    # def update(self, **kwargs) -> typing.Union[np.ndarray, typing.Mapping[str, np.ndarray]]:
    #     return self.forward(**kwargs)
    
    @abc.abstractmethod
    def forward(self, 
        *args, **kwargs
    ) -> typing.Union[np.ndarray, typing.Mapping[str, np.ndarray]]:
        """Update and calculate the current batch of metrics

        Downstream metric class implementations should specify their `torch.Tensor` arguments
        in the forward implementations.

        Forwarding corresponds to calculating and return the metric 
        for the current batch [`torch.Tensor`]
        """        
        pass

    @abc.abstractmethod
    def compute(self, 
        metrics: typing.Sequence[typing.Union[np.ndarray, typing.Mapping[str, np.ndarray]]]
    ) -> typing.Union[np.ndarray, typing.Mapping[str, np.ndarray]]:
        """Compute a global metric [`np.ndarray`] by aggregating all input metric results        
        """
        pass