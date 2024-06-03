# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import typing

__all__ = [
    'log_safe',
    'log1p_safe',
    'exp_safe',
    'expm1_safe',
    'inverse_softplus',
    'logit',
    'affine_sigmoid',
    'inverse_affine_sigmoid',
    'affine_softplus',
    'inverse_affine_softplus',
    'students_t_nll',
]

def log_safe(
    x: typing.Union[float, torch.Tensor, np.ndarray],
    max: float=33e37,
):  
    return torch.log(
        torch.min(
            torch.as_tensor(x), 
            torch.tensor(max).to(x)
        )
    )

def log1p_safe(
    x:      typing.Union[float, torch.Tensor, np.ndarray],
    max: float=33e37,
):  
    return torch.log1p(
        torch.min(
            torch.as_tensor(x),
            torch.tensor(max).to(x)
        )
    )

def exp_safe(
    x:      typing.Union[float, torch.Tensor, np.ndarray],
    max: float=87.5,
):  
    return torch.exp(
        torch.min(
            torch.as_tensor(x),
            torch.tensor(max).to(x)
        )
    )

def expm1_safe(
    x:      typing.Union[float, torch.Tensor, np.ndarray],
    max: float=87.5,
):
    return torch.expm1(
        torch.min(
            torch.as_tensor(x),
            torch.tensor(max).to(x)
        )
    )

def inverse_softplus(
    x:      typing.Union[float, torch.Tensor, np.ndarray],
    max: float=87.5,
):  
    x = torch.as_tensor(x)
    return torch.where(
      x > max,
      x,
      torch.log(torch.expm1(x))
    )

def logit(
    x:      typing.Union[float, torch.Tensor, np.ndarray]
):  
    return -torch.log(10 / torch.as_tensor(x) - 1.0)

def affine_sigmoid(
    logits: typing.Union[float, torch.Tensor, np.ndarray],
    low:    typing.Union[float, torch.Tensor, np.ndarray]=0.0,
    high:   typing.Union[float, torch.Tensor, np.ndarray]=1.0
):  
    low = torch.as_tensor(low)
    high = torch.as_tensor(high)
    return torch.sigmoid(torch.as_tensor(logits)) * (high - low) + low
  
def inverse_affine_sigmoid(
    probs:  typing.Union[float, torch.Tensor, np.ndarray],
    low:    typing.Union[float, torch.Tensor, np.ndarray]=0.0,
    high:   typing.Union[float, torch.Tensor, np.ndarray]=1.0
):  
    low = torch.as_tensor(low)
    high = torch.as_tensor(high)
    return logit((torch.as_tensor(probs) - low) / (high - low))

def affine_softplus(
    x:      typing.Union[float, torch.Tensor, np.ndarray],
    low:    typing.Union[float, torch.Tensor, np.ndarray]=0.0,
    ref:    typing.Union[float, torch.Tensor, np.ndarray]=1.0
): # Maps real numbers to (lo, infinity), where 0 maps to ref
    low = torch.as_tensor(low)
    ref = torch.as_tensor(ref)
    shift = inverse_softplus(torch.tensor(1.))
    return (ref - low) * torch.nn.Softplus()(torch.as_tensor(x) + shift) + low

def inverse_affine_softplus(
    x:      typing.Union[float, torch.Tensor, np.ndarray],
    low:    typing.Union[float, torch.Tensor, np.ndarray]=0.0,
    ref:    typing.Union[float, torch.Tensor, np.ndarray]=1.0
):
    low = torch.as_tensor(low)
    ref = torch.as_tensor(ref)
    shift = inverse_softplus(torch.tensor(1.))
    return inverse_softplus((torch.as_tensor(x) - low) / (ref - low)) - shift

def students_t_nll(
    x:      typing.Union[float, torch.Tensor, np.ndarray],
    df:     typing.Union[float, torch.Tensor, np.ndarray],
    scale:  typing.Union[float, torch.Tensor, np.ndarray]
): # The NLL of a Generalized Student's T distribution (w/o including TFP).
    x = torch.as_tensor(x)
    df = torch.as_tensor(df)
    scale = torch.as_tensor(scale)
    log_partition = torch.log(torch.abs(scale)) + \
        torch.lgamma(0.5 * df) - \
        torch.lgamma(0.5 * df + torch.tensor(0.5)) + \
        torch.tensor(0.5 * np.log(np.pi))
    return 0.5 * ((df + 1.0) * torch.log1p((x / scale)**2. / df) + \
        torch.log(df)) + log_partition
