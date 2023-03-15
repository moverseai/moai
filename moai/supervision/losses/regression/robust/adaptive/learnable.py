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

from moai.supervision.losses.regression.robust.adaptive.distribution import Distribution
from moai.supervision.losses.regression.robust.adaptive import util
from moai.utils.arguments.common import (
  assert_non_negative,
  assert_numeric,
)

import numpy as np
import torch
import typing
import logging

log = logging.getLogger(__name__)

__all__ = [
    'Barron',
    'Student',
]

class Barron(Distribution):
    def __init__(self,
        count:          int,
        alpha_range:    typing.Union[float, typing.Tuple[float, float]]=[0.001, 1.999],
        scale_low:      float=1e-5,
        scale_init:     float=1.0,
        alpha_init:     typing.Optional[float]=None,
    ):
      super().__init__()
      alpha_range = sorted(alpha_range)
      assert_numeric(log, 'alpha low', alpha_range[0], 0.0, 2.0)
      assert_numeric(log, 'alpha high', alpha_range[1], 0.0, 2.0)
      assert_non_negative(log, 'scale low', scale_low)
      if alpha_range[0] == alpha_range[1]: # constant init alpha
          log.info(f"Barron loss alpha set to fixed: {alpha_range[0]}.")
          self.register_buffer('fixed_alpha',
                                    torch.tensor(
              alpha_range[0])[np.newaxis, np.newaxis].repeat(1, count)
            )
          self.get_alpha = lambda: self.fixed_alpha
      else: # learnable alpha
          alpha = util.inverse_affine_sigmoid(
              np.mean(alpha_range) if alpha_init is None else alpha_init,
              low=alpha_range[0], high=alpha_range[1]
          )
          self.register_parameter('alpha', torch.nn.Parameter(
                  alpha.clone().detach().float()[np.newaxis, np.newaxis].repeat(1, count),
                  requires_grad=True
              )
          )
          self.get_alpha = lambda: util.affine_sigmoid(
              self.alpha, low=alpha_range[0], high=alpha_range[1]
          )
      if scale_low == scale_init: # fix `scale` to be a constant.
        log.info(f"Barron loss scale set to fixed: {scale_init}.")
        self.register_buffer('fixed_scale', 
          torch.tensor(scale_init)[np.newaxis, np.newaxis].float().repeat(1, count)
        )
        self.get_scale = lambda: self.fixed_scale
      else: # learnable scale
        self.register_parameter('scale', torch.nn.Parameter(
                # torch.zeros((1, count)).float(),
                torch.tensor(scale_init)[np.newaxis, np.newaxis].float().repeat(1, count),
           requires_grad=True
            )
        )
        self.get_scale = lambda: util.affine_softplus(
            self.scale, low=scale_low, ref=scale_init
        )
      #TODO: use parameterizations instead of lambdas

    def forward(self, 
        gt:   torch.Tensor,
        pred: torch.Tensor
    ) -> torch.Tensor:
        return super().forward(
            pred.flatten(1) - gt.flatten(1), 
            self.get_alpha(), 
            self.get_scale()
        )

class Student(torch.nn.Module):
    def __init__(self,
        count:      int,
        scale_low:  float=1e-5,
        scale_init: float=1.0,
    ):
      super(Student, self).__init__()
      assert_non_negative(log, "scale low", scale_low)
      self.register_parameter('log_df', torch.nn.Parameter(
          torch.zeros((1, count)).float(),
          requires_grad=True
      ))
      self.df = lambda: torch.exp(self.log_df)
      if scale_low == scale_init: # fix `scale` to be a constant.
          self.register_buffer('scale', 
            torch.tensor(scale_init).float()[np.newaxis, np.newaxis].repeat(1, count)
          )
          self.get_scale = lambda: self.scale
      else: # learnable scale
          self.register_parameter('scale', torch.nn.Parameter(
                torch.zeros((1, count)).float(), requires_grad=True
          ))
          self.get_scale = lambda: util.affine_softplus(
              self.scale, low=scale_low, ref=scale_init
          )
      #NOTE: explore parameterizations
      
    def lossfun(self, 
        gt:   torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        return util.students_t_nll(
          pred.flatten(1) - gt.flatten(1),
          self.df(), 
          self.scale()
        )
