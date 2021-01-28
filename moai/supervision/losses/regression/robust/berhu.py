from moai.monads.utils.spatial import expand_spatial_dims

import torch

__all__ = ["Berhu"]

class Berhu(torch.nn.Module):
    r"""Implements the (adaptive) berHu error function.

    ??? note "Berhu Loss"
        <p align="center"><img src="https://render.githubusercontent.com/render/math?math=%5Chuge%5Cbegin%7Bequation%7D%5Cmathcal%7BB%7D%28x%29%20%3D%20%5Cbegin%7Bcases%7D%20|x|%20%26%20|x|%20%5Cleq%20c%2C%20%5C%5C%20%5Cfrac%7Bx^2%20%2B%20c^2%7D%7B2c%7D%20%26%20|x|%20%3E%20c.%20%5C%5C%20%5Cend%7Bcases%7D%20%5Cend%7Bequation%7D"/></p>

    ??? note "Adaptive Threshold"
        <p align="center"><img src="https://render.githubusercontent.com/render/math?math=%5Chuge%20c%20%3D%20%5Cfrac%7B1%7D%7B5%7D%20%5Cmax_i%28|%5Ctilde%7By%7D_i%20-%20y_i|%29"/></p>

    ??? cite "Papers"
        [![Paper](https://img.shields.io/static/v1?label=1606.00373&message=Deeper Depth Prediction with Fully Convolutional Residual Networks&color=1c1ca2&logo=arxiv&style=plastic)](https://arxiv.org/pdf/1606.00373.pdf)

        [![Paper](https://img.shields.io/static/v1?label=1207.6868&message=The BerHu penalty and the grouped effect&color=1c1ca2&logo=arxiv&style=plastic)](https://arxiv.org/pdf/1207.6868.pdf)
 
    ???+ example "Configuration"
        === "Main Entry"
            ```yaml
            - model/supervision/losses/regression/robust: berhu
            ```
        === "Parameters"
            ```yaml            
            threshold: 1.0
            adaptive: false # toggle adaptive threshold, ignores given threshold
            image_wide: false # adaptive threshold calculated per batch item (true) or per batch (false)
            ```
        === "DML"
            ```yaml
            model:
              supervision:
                losses:
                  berhu:
                    threshold: 1.0
                    adaptive: false
                    image_wide: false
            ```
        === "Graph"
            ```yaml
            model:
              supervision:
                losses:
                  berhu:
                    gt: [groundtruth_tensor_name] # the name of the groundtruth tensor
                    pred: [prediction_tensor_name] # the name of the predicted tensor
                    out: [berhu_loss] # optional, will be 'berhu' if omitted
            ```
       
    Arguments:
        threshold (float, optional):
            Sets the threshold that switches between the L1 and L2 loss.       
        adaptive (boolean, optional):
            If true, the threshold is calculated dynamically for each batch. **Default value: False.**
        image_wide (boolean, optional): 
            If true, the adaptive threshold is calculated for each batch item. **Default value: False.**        

    !!! important
        Can be helpful when dealing with heavy-tailed distribution of predicted values.

    ??? info "Repositories"
        [![Repo1](https://github-readme-stats.vercel.app/api/pin/?username=iro-cp&repo=FCRN-DepthPrediction&hide_border=true&title_color=1c1ca2&show_owner=true)](https://github.com/iro-cp/FCRN-DepthPrediction){: align=left }
        [![Repo2](https://github-readme-stats.vercel.app/api/pin/?username=dontLoveBugs&repo=FCRN_pytorch&hide_border=true&title_color=1c1ca2&show_owner=true)](https://github.com/dontLoveBugs/FCRN_pytorch){: align=right }

    *[berHu]: Reverse Huber

    """
    def __init__(self,
        threshold:      float,
        adaptive:       bool=False,
        image_wide:     bool=False,
    ):
        super(Berhu, self).__init__()
        self.threshold = threshold
        self.adaptive = adaptive
        self.image_wide = image_wide

    def _get_max(self, error: torch.Tensor) -> torch.Tensor:
        if self.image_wide:
            values, indices = torch.max(error.view(error.shape[0], -1), dim=1)
            expanded = expand_spatial_dims(values, error)
            if error.shape[1] != 1:
                expanded = expanded.expand(error.shape[0], error.shape[1], 1)
            return expanded
        else:
            return torch.max(error)

    def forward(self, 
        gt:         torch.Tensor,
        pred:       torch.Tensor,
        weights:    torch.Tensor=None,       # float tensor
        mask:       torch.Tensor=None,       # byte tensor
    ) -> torch.Tensor:        
        L1 = torch.abs(gt - pred)
        if weights is not None:
            L1 = L1 * weights
        if mask is not None:
            L1 = L1[mask]
        threshold = self.threshold if not self.adaptive else 0.2 * self._get_max(L1)
        if len(threshold.shape) > 1:
            L1 = L1.view(*threshold.shape[:2], -1)
        berhu = torch.where(
            L1 <= threshold,
            L1,
            (L1 ** 2 + threshold ** 2) / (2.0 * threshold)
        )
        return berhu.view_as(gt)