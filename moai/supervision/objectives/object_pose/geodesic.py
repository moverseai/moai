import torch


class Geodesic(torch.nn.Module):
    r"""Implements the geodesic error function.

    ??? note "Geodesic Loss"
         <p align="center"><img src="https://render.githubusercontent.com/render/math?math=%5Chuge%5Cbegin%7Bequation%7D%5Cmathcal%7Bd%7D%28R_1%2CR_2%29%20%3D%20%5Carccos%5Cfrac%7Btrace%28R_1R_2^T%29%20-%201%7D%7B2%7D%5Cend%7Bequation%7D"/></p>

    ??? cite "Papers"
        [![Paper](https://img.shields.io/static/v1?label=1606.00373&message=On the Continuity of Rotation Representations in Neural Networks&color=1c1ca2&logo=arxiv&style=plastic)](https://arxiv.org/pdf/1812.07035.pdf)

        [![Paper](https://img.shields.io/static/v1?label=1207.6868&message=3D Pose Regression using Convolutional Neural Networks&color=1c1ca2&logo=arxiv&style=plastic)](https://arxiv.org/pdf/1708.05628.pdf)

    ???+ example "Configuration"
        === "Main Entry"
            ```yaml
            - model/supervision/losses/object_pose: geodesic
            ```
        === "Parameters"
            _The geodesic loss has no parameters currently._
        === "Graph"
            ```yaml
            model:
              supervision:
                losses:
                  geodesic:
                    gt: [groundtruth_tensor_name] # the name of the groundtruth tensor
                    pred: [prediction_tensor_name] # the name of the predicted tensor
                    out: [geodesic_loss] # optional, will be 'geodesic' if omitted
            ```


    !!! important
        Both matrices should be orthogonal.

    """

    def __init__(self):
        super(Geodesic, self).__init__()

    def forward(
        self,
        pred: torch.Tensor,  # b x 3 x 3
        gt: torch.Tensor,  # b x 3 x 3
    ) -> torch.Tensor:
        return _compute_geodesic_loss(gt, pred)


def _compute_geodesic_loss(gt_r_matrix, out_r_matrix):
    batch = gt_r_matrix.shape[0]
    m = torch.bmm(gt_r_matrix, out_r_matrix.transpose(1, 2))  # [B, 3, 3]
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1.0) / 2.0
    cos = torch.min(cos, torch.ones(batch).to(gt_r_matrix))
    cos = torch.max(cos, torch.ones(batch).to(gt_r_matrix) * -1.0)
    theta = torch.acos(cos)
    return theta
