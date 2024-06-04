import typing

import numpy as np
import torch


# NOTE: implementation derived from: https://github.com/YvanYin/VNL_Monocular_Depth_Prediction
class VirtualNormal(torch.nn.Module):
    def __init__(
        self,
        delta_cos: float = 0.867,
        delta_diff_x: float = 0.01,
        delta_diff_y: float = 0.01,
        delta_diff_z: float = 0.01,
        delta_z: float = 0.0001,
        sample_ratio: float = 0.15,
        ignore_perc: float = 0.0,
    ):
        super(VirtualNormal, self).__init__()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio
        self.ignore_perc = ignore_perc

    def select_index(self, width: int, height: int) -> typing.Dict[str, torch.Tensor]:
        num = width * height
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        p1_x = p1 % width
        p1_y = (p1 / width).astype(np.int32)

        p2_x = p2 % width
        p2_y = (p2 / width).astype(np.int32)

        p3_x = p3 % width
        p3_y = (p3 / width).astype(np.int32)

        p123 = {
            "p1_x": p1_x,
            "p1_y": p1_y,
            "p2_x": p2_x,
            "p2_y": p2_y,
            "p3_x": p3_x,
            "p3_y": p3_y,
        }
        return p123

    def form_pw_groups(
        self, p123: typing.Dict[str, torch.Tensor], pw: torch.Tensor
    ) -> torch.Tensor:
        """
        Form 3D points groups, with 3 points in each grouup.
        :param p123: points index
        :param pw: 3D points
        :return:
        """
        p1_x = p123["p1_x"]
        p1_y = p123["p1_y"]
        p2_x = p123["p2_x"]
        p2_y = p123["p2_y"]
        p3_x = p123["p3_x"]
        p3_y = p123["p3_y"]

        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat(
            [
                pw1[:, :, :, np.newaxis],
                pw2[:, :, :, np.newaxis],
                pw3[:, :, :, np.newaxis],
            ],
            3,
        )
        return pw_groups

    def filter_mask(
        self,
        p123: torch.Tensor,
        gt_xyz: torch.Tensor,
        delta_cos: float = 0.867,
        delta_diff_x: float = 0.005,
        delta_diff_y: float = 0.005,
        delta_diff_z: float = 0.005,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        # NOTE: ignore co-linear
        pw_diff = torch.cat(
            [
                pw12[:, :, :, np.newaxis],
                pw13[:, :, :, np.newaxis],
                pw23[:, :, :, np.newaxis],
            ],
            3,
        )  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(
            0, 2, 1
        )  # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.view(
            m_batchsize * groups, -1, index
        )  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(
            q_norm.view(m_batchsize * groups, index, 1),
            q_norm.view(m_batchsize * groups, 1, index),
        )  # []
        energy = torch.bmm(
            proj_query, proj_key
        )  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1)
        mask_cos = (
            torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3
        )  # igonre
        mask_cos = mask_cos.view(m_batchsize, groups)
        # NOTE: ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3
        # NOTE: ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near
        return mask, pw

    def select_points_groups(
        self, gt_xyz: torch.Tensor, pred_xyz: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        pw_gt = gt_xyz.transpose(1, 3).transpose(1, 2)
        pw_pred = pred_xyz.transpose(1, 3).transpose(1, 2)
        B, C, H, W = gt_xyz.shape
        p123 = self.select_index(W, H)
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        mask, pw_groups_gt = self.filter_mask(
            p123,
            pw_gt,
            delta_cos=0.867,
            delta_diff_x=0.005,
            delta_diff_y=0.005,
            delta_diff_z=0.005,
        )  # [b, n, 3, 3]
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001
        mask_broadcast = mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2)
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(1, -1, 3, 3)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(
        self,
        gt: torch.Tensor,  # [BCHW] /w C = 3 (XYZ)
        pred: torch.Tensor,  # [BCHW] /w C = 3 (XYZ)
        weights: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Virtual normal loss.
        :param pred_depth: predicted depth map, [B,W,H,C]
        :param data: target label, ground truth depth, [B, W, H, C], padding region [padding_up, padding_down]
        :return:
        """
        if weights is not None:
            gt_xyz = weights * gt
            pred_xyz = weights * pred
        elif mask is not None:
            gt_xyz = gt.clone()
            pred_xyz = pred.clone()
            gt_xyz[~mask] = 0.0
            pred_xyz[~mask] = 0.0
        else:
            gt_xyz = gt
            pred_xyz = pred

        gt_points, dt_points = self.select_points_groups(gt, pred)

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
        dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = dt_norm == 0.0
        gt_mask = gt_norm == 0.0
        dt_mask = dt_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        dt_mask *= 0.01
        gt_mask *= 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm
        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if self.ignore_perc:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * self.ignore_perc) :]
        loss = torch.mean(loss)
        return loss
