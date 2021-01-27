from moai.supervision.losses.distribution.kl_divergence import KL
from moai.supervision.losses.distribution.std_kl import StandardNormalKL
from moai.supervision.losses.distribution.lambda_divergence import Lambda, JS
from moai.supervision.losses.distribution.variance import VarianceRegularization

__all__ = [
    "KL",
    "Lambda",
    "JS",
    "StandardNormalKL",
    "VarianceRegularization",
]