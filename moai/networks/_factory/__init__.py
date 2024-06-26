from moai.networks.lightning.factory.autoencoder import Autoencoder
from moai.networks.lightning.factory.cascade import Cascade
from moai.networks.lightning.factory.hrnet import HRNet
from moai.networks.lightning.factory.multibranch import MultiBranch
from moai.networks.lightning.factory.stacked_hourglass import StackedHourglass
from moai.networks.lightning.factory.unet import UNet
from moai.networks.lightning.factory.vae import VariationalAutoencoder

try:
    from moai.networks.lightning.factory.onnx import Onnx
except ImportError:
    print(
        "ONNX is not available, please make sure you have installed all the dependencies (e.g. onnx, onnx-runtime)."
    )
from moai.networks.lightning.factory.gan import (
    GenerativeAdversarialNetwork,
    NullGenerativeAdversarialNetwork,
)

__all__ = [
    "Autoencoder",
    "HRNet",
    "MultiBranch",
    "StackedHourglass",
    "VariationalAutoencoder",
    "Cascade",
    "UNet",
    "Onnx",
    "GenerativeAdversarialNetwork",
    "NullGenerativeAdversarialNetwork",
]
