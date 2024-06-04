import functools
import typing

import omegaconf.omegaconf
import toolz
import torch

import moai.nn.convolution as mic
import moai.nn.linear as milin
from moai.nn.sampling.spatial.downsample import make_downsample

__all__ = ["SqueezeExcite"]


# NOTE: from https://github.com/xvjiarui/GCNet/blob/029db5407dc27147eb1d41f62b09dfed8ec88837/mmdet/ops/gcb/context_block.py#L64
class Attention2d(torch.nn.Module):
    def __init__(
        self,
        features: int,
    ):
        super(Attention2d, self).__init__()
        self.conv = torch.nn.Conv2d(features, 1, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = x.view(b, c, h * w).unsqueeze(1)
        mask = self.conv(x)
        mask = torch.nn.functional.softmax(mask.view(b, 1, h * w), dim=-1).unsqueeze(-1)
        context = torch.matmul(y, mask)
        return context.view(b, c, 1, 1)


__SQUEEZE__ = {
    "average1d": functools.partial(make_downsample, downscale_type="global1d_avg"),
    "average2d": functools.partial(make_downsample, downscale_type="global2d_avg"),
    "average3d": functools.partial(make_downsample, downscale_type="global3d_avg"),
    "attention2d": Attention2d,
    "none": torch.nn.Identity,
}


def make_channel_conv(
    type: str,
    in_features: int,
    out_features: int,
    operation_params: dict,
    activation_type: str,
    activation_params: dict,
):
    return mic.make_conv_block(
        block_type=type,
        convolution_type=type,
        in_features=in_features,
        out_features=out_features,
        activation_type=activation_type,
        convolution_params=toolz.merge(
            operation_params, {"kernel_size": 1, "padding": 0, "stride": 1}
        ),
        activation_params=toolz.merge({"inplace": True}, activation_params),
    )


def make_channel_linear(
    type: str,
    in_features: int,
    out_features: int,
    operation_params: dict,
    activation_type: str,
    activation_params: dict,
):
    return milin.make_linear_block(
        block_type=type,
        linear_type=type,
        in_features=in_features,
        out_features=out_features,
        activation_type=activation_type,
        linear_params=operation_params,
        activation_params=toolz.merge({"inplace": True}, activation_params),
    )


__OPERATIONS__ = {
    "conv2d": functools.partial(make_channel_conv, type="conv2d"),
    "linear": functools.partial(make_channel_linear, type="linear"),
}


def make_channel(
    operation_type: str,
    activation_type: str,
    in_features: int,
    operation_params: dict = None,
    activation_params: dict = None,
    ratio: typing.Union[float, int] = 0.5,
):
    inter_features = (
        int(in_features * ratio) if isinstance(ratio, float) else in_features // ratio
    )
    return torch.nn.Sequential(
        __OPERATIONS__[operation_type](
            in_features=in_features,
            out_features=inter_features,
            operation_params=operation_params or {},
            activation_type=activation_type or "none",
            activation_params=activation_params or {},
        ),
        __OPERATIONS__[operation_type](
            in_features=inter_features,
            out_features=in_features,
            operation_params=operation_params or {},
            activation_type="none",
            activation_params={},
        ),
        (
            torch.nn.Unflatten(1, (in_features, 1, 1))
            if "linear" in operation_type
            else torch.nn.Identity()
        ),
    )


def make_spatial(
    operation_type: str,
    activation_type: str,
    in_features: int,
    operation_params: dict = None,
    activation_params: dict = None,
    ratio: float = 1.0,
):
    return make_channel_conv(
        type=operation_type,
        in_features=in_features,
        out_features=1,
        operation_params=operation_params or {},
        activation_type=activation_type,
        activation_params=activation_params or {},
    )


__EXCITE__ = {
    "channel": make_channel,
    "spatial": make_spatial,
    # TODO: 'channel_spatial': #NOTE: with max of channel and spatial, see https://github.com/ai-med/squeeze_and_excitation/blob/acdd26e7e3956b8e3d3b32663a784ebd64c844dd/squeeze_and_excitation/squeeze_and_excitation_3D.py#L119
}


# NOTE: it is an activation
# TODO: check with bias as well: https://github.com/JYPark09/SENet-PyTorch/blob/6f1eae93256e5181baea8d5102473c6cba6500fa/network.py#L6
class SqueezeExcite(torch.nn.Module):
    def __init__(
        self,
        features: int,
        squeeze: omegaconf.DictConfig,  # type: 'averageXd', # one of ['averageXd', 'attentionXd']
        excite: omegaconf.DictConfig,  # delta: activation_{type|params}, # ratio \in [0, 1], # operation_{type|params}: one of ['convXd', 'linear']
        mode: str = "mul",
    ):
        super(SqueezeExcite, self).__init__()
        self.squeeze = __SQUEEZE__[squeeze["type"]](features=features)
        self.excite = __EXCITE__[excite["type"]](
            operation_type=excite["operation"]["type"],
            activation_type=toolz.get_in(["activation", "type"], excite, "none"),
            in_features=features,
            operation_params=toolz.get_in(["operation", "params"], excite, {}),
            activation_params=toolz.get_in(["activation", "params"], excite, {}),
            ratio=excite["ratio"],
        )
        self.mode = getattr(torch, mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = self.squeeze(x)
        excited = self.excite(squeezed)
        return self.mode(torch.sigmoid(excited), x)
