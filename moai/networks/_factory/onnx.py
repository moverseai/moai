import logging
import typing

import numpy as np
import omegaconf.omegaconf
import toolz
import torch

import moai.networks.lightning as minet

__all__ = ["Onnx"]

log = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    log.warning(
        "onnxruntime not installed. Please install onnxruntime to use ONNX models."
    )


class Onnx(minet.FeedForward):
    def __init__(
        self,
        configuration: omegaconf.DictConfig,
        data: omegaconf.DictConfig = None,
        parameters: omegaconf.DictConfig = None,
        feedforward: omegaconf.DictConfig = None,
        monads: omegaconf.DictConfig = None,
        supervision: omegaconf.DictConfig = None,
        validation: omegaconf.DictConfig = None,
        visualization: omegaconf.DictConfig = None,
        export: omegaconf.DictConfig = None,
    ):
        super(Onnx, self).__init__(
            feedforward=feedforward,
            monads=monads,
            data=data,
            validation=validation,
            parameters=parameters,
            visualization=visualization,
            export=export,
            supervision=supervision,
        )
        self.input_kvp = configuration.input
        self.output_list = configuration.output
        options = ort.SessionOptions()
        options.enable_profiling = True
        self.session = ort.InferenceSession(
            configuration.path,
            sess_options=options,
        )
        log.info(f"Successfully loaded {configuration.path} model")
        self.retreived_inputs = self.session.get_inputs()

    def to_numpy(
        self,
        tensor: torch.Tensor,
    ) -> np.array:
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    def forward(
        self, td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        pred = self.session.run(
            self.output_list,
            toolz.valmap(lambda v: self.to_numpy(td[v]), self.input_kvp),
        )
        for k, t in zip(self.output_list, pred):
            td[k] = torch.from_numpy(t)
        return td
