import logging
import typing

import cv2
import numpy as np
import torch

# NOTE: https://github.com/BoChenYS/BPnP
# NOTE: https://arxiv.org/pdf/1909.06043.pdf


log = logging.getLogger(__name__)

try:
    import kornia as kn
except ImportError:
    log.error("Kornia is required for BPnP, please install it before proceeding.")

__all__ = ["BPnP"]


def _batch_project(P, pts3d, K, angle_axis=True):
    n = pts3d.size(0)
    bs = P.size(0)
    device = P.device
    pts3d_h = torch.cat((pts3d, torch.ones(n, 1, device=device)), dim=-1)
    if angle_axis:
        R_out = kn.angle_axis_to_rotation_matrix(P[:, 0:3].reshape(bs, 3))
        PM = torch.cat((R_out[:, 0:3, 0:3], P[:, 3:6].reshape(bs, 3, 1)), dim=-1)
    else:
        PM = P
    pts3d_cam = pts3d_h.matmul(PM.transpose(-2, -1))
    pts2d_proj = pts3d_cam.matmul(K.t())
    S = pts2d_proj[:, :, 2].reshape(bs, n, 1)
    pts2d_pro = pts2d_proj[:, :, 0:2] / (S + 1e-12)
    return pts2d_pro


def _get_coefs(P_6d, pts3d, K):
    device = P_6d.device
    n = pts3d.size(0)
    m = P_6d.size(-1)
    coefs = torch.zeros(n, 2, m, device=device)
    torch.set_grad_enabled(True)
    y = P_6d.repeat(n, 1)
    proj = _batch_project(y, pts3d, K).squeeze()
    vec = torch.diag(torch.ones(n, device=device).float())
    for k in range(2):
        torch.set_grad_enabled(True)
        y_grad = torch.autograd.grad(
            proj[:, :, k], y, vec, retain_graph=True, create_graph=True
        )
        coefs[:, k, :] = -2.0 * y_grad[0].clone()
    return coefs


class BPnPFunction_fast(torch.autograd.Function):
    """
    BPnP_fast is the efficient version of the BPnP class which ignores the higher order dirivatives through the coefs' graph. This sacrifices
    negligible gradient accuracy yet saves significant runtime.
    INPUTS:
    pts2d - the 2D keypoints coordinates of size [batch_size, num_keypoints, 2]
    pts3d - the 3D keypoints coordinates of size [num_keypoints, 3]
    K     - the camera intrinsic matrix of size [3, 3]
    OUTPUT:
    P_6d  - the 6 DOF poses of size [batch_size, 6], where the first 3 elements of each row are the angle-axis rotation
    vector (Euler vector) and the last 3 elements are the translation vector.
    NOTE:
    This BPnP function assumes that all sets of 2D points in the mini-batch correspond to one common set of 3D points.
    For situations where pts3d is also a mini-batch, use the BPnP_m3d class.
    """

    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None):
        bs = pts2d.size(0)
        n = pts2d.size(1)
        device = pts2d.device
        K_np = np.array(K.detach().cpu())
        P_6d = torch.zeros(bs, 6, device=device)

        for i in range(bs):
            pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape(
                (n, 1, 2)
            )
            pts3d_i_np = np.ascontiguousarray(pts3d[i].detach().cpu()).reshape((n, 3))
            if ini_pose is None:
                _, rvec0, T0 = cv2.solvePnP(
                    objectPoints=pts3d_i_np,
                    imagePoints=pts2d_i_np,
                    cameraMatrix=K_np,
                    distCoeffs=None,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    useExtrinsicGuess=False,
                )
            else:
                rvec0 = np.array(ini_pose[i, 0:3].cpu().reshape(3, 1))
                T0 = np.array(ini_pose[i, 3:6].cpu().reshape(3, 1))
            _, rvec, T = cv2.solvePnP(
                objectPoints=pts3d_i_np,
                imagePoints=pts2d_i_np,
                cameraMatrix=K_np,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec0,
                tvec=T0,
            )
            _, rvec, T = cv2.solvePnP(
                objectPoints=pts3d_i_np,
                imagePoints=pts2d_i_np,
                cameraMatrix=K_np,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec0,
                tvec=T0,
            )
            angle_axis = torch.tensor(rvec, device=device, dtype=torch.float).reshape(
                1, 3
            )
            T = torch.tensor(T, device=device, dtype=torch.float).reshape(1, 3)
            P_6d[i, :] = torch.cat((angle_axis, T), dim=-1)

        ctx.save_for_backward(pts2d, P_6d, pts3d, K)
        return P_6d

    @staticmethod
    def backward(ctx, grad_output):
        pts2d, P_6d, pts3d, K = ctx.saved_tensors
        device = pts2d.device
        bs = pts2d.size(0)
        n = pts2d.size(1)
        m = 6

        grad_x = torch.zeros_like(pts2d)
        grad_z = torch.zeros_like(pts3d)
        grad_K = torch.zeros_like(K)

        for i in range(bs):
            J_fy = torch.zeros(m, m, device=device)
            J_fx = torch.zeros(m, 2 * n, device=device)
            J_fz = torch.zeros(m, 3 * n, device=device)
            J_fK = torch.zeros(m, 9, device=device)

            pts2d_flat = pts2d[i].clone().reshape(-1).detach().requires_grad_()
            P_6d_flat = P_6d[i].clone().reshape(-1).detach().requires_grad_()
            pts3d_flat = pts3d[i].clone().reshape(-1).detach().requires_grad_()
            coefs = _get_coefs(
                P_6d[i].reshape(1, 6), pts3d_flat.reshape(n, 3), K
            ).detach()
            K_flat = K.clone().reshape(-1).detach().requires_grad_()

            for j in range(m):
                torch.set_grad_enabled(True)
                if j > 0:
                    pts2d_flat.grad.zero_()
                    P_6d_flat.grad.zero_()
                    pts3d_flat.grad.zero_()
                    K_flat.grad.zero_()

                R = kn.angle_axis_to_rotation_matrix(P_6d_flat[0 : m - 3].reshape(1, 3))

                P = torch.cat(
                    (R[0, 0:3, 0:3].reshape(3, 3), P_6d_flat[m - 3 : m].reshape(3, 1)),
                    dim=-1,
                )
                KP = torch.mm(K_flat.reshape(3, 3), P)
                pts2d_i = pts2d_flat.reshape(n, 2).transpose(0, 1)
                pts3d_i = torch.cat(
                    (pts3d_flat.reshape(n, 3), torch.ones(n, 1, device=device)), dim=-1
                ).t()
                proj_i = KP.mm(pts3d_i)
                Si = proj_i[2, :].reshape(1, n)

                r = pts2d_i * Si - proj_i[0:2, :]
                coef = coefs[:, :, j].transpose(0, 1)  # size: [2,n]
                fj = (coef * r).sum()
                fj.backward()
                J_fy[j, :] = P_6d_flat.grad.clone()
                J_fx[j, :] = pts2d_flat.grad.clone()
                J_fz[j, :] = pts3d_flat.grad.clone()
                J_fK[j, :] = K_flat.grad.clone()

            inv_J_fy = torch.inverse(J_fy)

            J_yx = (-1) * torch.mm(inv_J_fy, J_fx)
            J_yz = (-1) * torch.mm(inv_J_fy, J_fz)
            J_yK = (-1) * torch.mm(inv_J_fy, J_fK)

            grad_x[i] = grad_output[i].reshape(1, m).mm(J_yx).reshape(n, 2)
            grad_z += grad_output[i].reshape(1, m).mm(J_yz).reshape(n, 3)
            grad_K += grad_output[i].reshape(1, m).mm(J_yK).reshape(3, 3)

        return grad_x, grad_z, grad_K, None


class BPnPFunction_m3d(torch.autograd.Function):
    """
    BPnP_m3d supports mini-batch inputs of 3D keypoints, where the i-th set of 2D keypoints correspond to the i-th set of 3D keypoints.
    INPUTS:
    pts2d - the 2D keypoints coordinates of size [batch_size, num_keypoints, 2]
    pts3d - the 3D keypoints coordinates of size [batch_size, num_keypoints, 3]
    K     - the camera intrinsic matrix of size [3, 3]
    OUTPUT:
    P_6d  - the 6 DOF poses of size [batch_size, 6], where the first 3 elements of each row are the angle-axis rotation
    vector (Euler vector) and the last 3 elements are the translation vector.
    NOTE:
    For situations where all sets of 2D points in the mini-batch correspond to one common set of 3D points, use the BPnP class.
    """

    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None):
        bs = pts2d.size(0)
        n = pts2d.size(1)
        device = pts2d.device
        K_np = np.array(K.detach().cpu())
        P_6d = torch.zeros(bs, 6, device=device)

        for i in range(bs):
            pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape(
                (n, 1, 2)
            )
            pts3d_i_np = np.ascontiguousarray(pts3d[i].detach().cpu()).reshape((n, 3))
            if ini_pose is None:
                _, rvec0, T0 = cv2.solvePnP(
                    objectPoints=pts3d_i_np,
                    imagePoints=pts2d_i_np,
                    cameraMatrix=K_np,
                    distCoeffs=None,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    useExtrinsicGuess=False,
                )
            else:
                rvec0 = np.array(ini_pose[i, 0:3].cpu().reshape(3, 1))
                T0 = np.array(ini_pose[i, 3:6].cpu().reshape(3, 1))
            _, rvec, T = cv2.solvePnP(
                objectPoints=pts3d_i_np,
                imagePoints=pts2d_i_np,
                cameraMatrix=K_np,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec0,
                tvec=T0,
            )
            angle_axis = torch.tensor(rvec, device=device, dtype=torch.float).reshape(
                1, 3
            )
            T = torch.tensor(T, device=device, dtype=torch.float).reshape(1, 3)
            P_6d[i, :] = torch.cat((angle_axis, T), dim=-1)

        ctx.save_for_backward(pts2d, P_6d, pts3d, K)
        return P_6d

    @staticmethod
    def backward(ctx, grad_output):
        pts2d, P_6d, pts3d, K = ctx.saved_tensors
        device = pts2d.device
        bs = pts2d.size(0)
        n = pts2d.size(1)
        m = 6

        grad_x = torch.zeros_like(pts2d)
        grad_z = torch.zeros_like(pts3d)
        grad_K = torch.zeros_like(K)

        for i in range(bs):
            J_fy = torch.zeros(m, m, device=device)
            J_fx = torch.zeros(m, 2 * n, device=device)
            J_fz = torch.zeros(m, 3 * n, device=device)
            J_fK = torch.zeros(m, 9, device=device)

            torch.set_grad_enabled(True)
            pts2d_flat = pts2d[i].clone().reshape(-1).detach().requires_grad_()
            P_6d_flat = P_6d[i].clone().reshape(-1).detach().requires_grad_()
            pts3d_flat = pts3d[i].clone().reshape(-1).detach().requires_grad_()
            K_flat = K.clone().reshape(-1).detach().requires_grad_()

            for j in range(m):
                torch.set_grad_enabled(True)
                if j > 0:
                    pts2d_flat.grad.zero_()
                    P_6d_flat.grad.zero_()
                    pts3d_flat.grad.zero_()
                    K_flat.grad.zero_()

                R = kn.angle_axis_to_rotation_matrix(P_6d_flat[0 : m - 3].reshape(1, 3))

                P = torch.cat(
                    (R[0, 0:3, 0:3].reshape(3, 3), P_6d_flat[m - 3 : m].reshape(3, 1)),
                    dim=-1,
                )
                KP = torch.mm(K_flat.reshape(3, 3), P)
                pts2d_i = pts2d_flat.reshape(n, 2).transpose(0, 1)
                pts3d_i = torch.cat(
                    (pts3d_flat.reshape(n, 3), torch.ones(n, 1, device=device)), dim=-1
                ).t()
                proj_i = KP.mm(pts3d_i)
                Si = proj_i[2, :].reshape(1, n)

                r = pts2d_i * Si - proj_i[0:2, :]
                coefs = _get_coefs(
                    P_6d_flat.reshape(1, 6),
                    pts3d_flat.reshape(n, 3),
                    K_flat.reshape(3, 3),
                )
                coef = coefs[:, :, j].transpose(0, 1)  # size: [2,n]
                fj = (coef * r).sum()
                fj.backward()
                J_fy[j, :] = P_6d_flat.grad.clone()
                J_fx[j, :] = pts2d_flat.grad.clone()
                J_fz[j, :] = pts3d_flat.grad.clone()
                J_fK[j, :] = K_flat.grad.clone()

            inv_J_fy = torch.inverse(J_fy)

            J_yx = (-1) * torch.mm(inv_J_fy, J_fx)
            J_yz = (-1) * torch.mm(inv_J_fy, J_fz)
            J_yK = (-1) * torch.mm(inv_J_fy, J_fK)

            grad_x[i] = grad_output[i].reshape(1, m).mm(J_yx).reshape(n, 2)
            grad_z[i] = grad_output[i].reshape(1, m).mm(J_yz).reshape(n, 3)
            grad_K += grad_output[i].reshape(1, m).mm(J_yK).reshape(3, 3)

        return grad_x, grad_z, grad_K, None


class BPnPFunction(torch.autograd.Function):
    r"""Back-propagatable PnP
    Arguments:
        pts2d - the 2D keypoints coordinates of size [batch_size, num_keypoints, 2]
        pts3d - the 3D keypoints coordinates of size [num_keypoints, 3]
        K     - the camera intrinsic matrix of size [3, 3]
    Returns:
        P_6d  - the 6 DOF poses of size [batch_size, 6], where the first 3 elements of each row are the angle-axis rotation
        vector (Euler vector) and the last 3 elements are the translation vector.
    NOTE: This bpnp function assumes that all sets of 2D points in the mini-batch correspond to one common set of 3D points.
    """

    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None):
        bs = pts2d.size(0)
        n = pts2d.size(1)
        device = pts2d.device
        pts3d_np = np.array(pts3d.detach().cpu())
        K_np = np.array(K.detach().cpu())
        P_6d = torch.zeros(bs, 6, device=device)

        try:
            distCoeffs = pnp.distCoeffs
        except:
            distCoeffs = np.zeros((8, 1), dtype="float32")

        for i in range(bs):
            pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape(
                (n, 1, 2)
            )
            if ini_pose is None:
                _, rvec0, T0 = cv2.solvePnP(
                    objectPoints=pts3d_np,
                    imagePoints=pts2d_i_np,
                    cameraMatrix=K_np,
                    distCoeffs=distCoeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    useExtrinsicGuess=False,
                )
            else:
                rvec0 = np.array(ini_pose[0, 0:3].cpu().reshape(3, 1))
                T0 = np.array(ini_pose[0, 3:6].cpu().reshape(3, 1))
            _, rvec, T = cv2.solvePnP(
                objectPoints=pts3d_np,
                imagePoints=pts2d_i_np,
                cameraMatrix=K_np,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec0,
                tvec=T0,
            )
            angle_axis = torch.Tensor(rvec).reshape(1, 3).float().to(device)
            T = torch.Tensor(T).reshape(1, 3).float().to(device)
            P_6d[i, :] = torch.cat((angle_axis, T), dim=-1)

        ctx.save_for_backward(pts2d, P_6d, pts3d, K)
        return P_6d

    @staticmethod
    def backward(ctx, grad_output):
        pts2d, P_6d, pts3d, K = ctx.saved_tensors
        device = pts2d.device
        bs = pts2d.size(0)
        n = pts2d.size(1)  # nof keypoints
        m = 6

        grad_x = torch.zeros_like(pts2d)
        grad_z = torch.zeros_like(pts3d)
        grad_K = torch.zeros_like(K)

        for i in range(bs):
            J_fy = torch.zeros(m, m, device=device)
            J_fx = torch.zeros(m, 2 * n, device=device)
            J_fz = torch.zeros(m, 3 * n, device=device)
            J_fK = torch.zeros(m, 9, device=device)

            torch.set_grad_enabled(True)
            pts2d_flat = pts2d[i].clone().reshape(-1).detach().requires_grad_()
            P_6d_flat = P_6d[i].clone().reshape(-1).detach().requires_grad_()
            pts3d_flat = pts3d.clone().reshape(-1).detach().requires_grad_()
            K_flat = K.clone().reshape(-1).detach().requires_grad_()

            for j in range(m):
                if j > 0:
                    pts2d_flat.grad.zero_()
                    P_6d_flat.grad.zero_()
                    pts3d_flat.grad.zero_()
                    K_flat.grad.zero_()

                R = kn.angle_axis_to_rotation_matrix(P_6d_flat[0 : m - 3].reshape(1, 3))

                P = torch.cat(
                    (R[0, 0:3, 0:3].reshape(3, 3), P_6d_flat[m - 3 : m].reshape(3, 1)),
                    dim=-1,
                )
                KP = torch.mm(K_flat.reshape(3, 3), P)
                pts2d_i = pts2d_flat.reshape(n, 2).transpose(0, 1)
                pts3d_i = torch.cat(
                    (pts3d_flat.reshape(n, 3), torch.ones(n, 1, device=device)), dim=-1
                ).t()
                proj_i = KP.mm(pts3d_i)
                Si = proj_i[2, :].reshape(1, n)

                r = pts2d_i * Si - proj_i[0:2, :]

                coefs = _get_coefs(
                    P_6d_flat.reshape(1, 6),
                    pts3d_flat.reshape(n, 3),
                    K_flat.reshape(3, 3),
                )

                coef = coefs[:, :, j].transpose(0, 1)  # size: [2,n]
                fj = (coef * r).sum()
                fj.backward()
                J_fy[j, :] = P_6d_flat.grad.clone()
                J_fx[j, :] = pts2d_flat.grad.clone()
                J_fz[j, :] = pts3d_flat.grad.clone()
                J_fK[j, :] = K_flat.grad.clone()

            inv_J_fy = torch.inverse(J_fy)

            J_yx = (-1.0) * torch.mm(inv_J_fy, J_fx)
            J_yz = (-1.0) * torch.mm(inv_J_fy, J_fz)
            J_yK = (-1.0) * torch.mm(inv_J_fy, J_fK)

            grad_x[i] = grad_output[i].reshape(1, m).mm(J_yx).reshape(n, 2)
            grad_z += grad_output[i].reshape(1, m).mm(J_yz).reshape(n, 3)
            grad_K += grad_output[i].reshape(1, m).mm(J_yK).reshape(3, 3)

        return grad_x, grad_z, grad_K, None


class BPnP(torch.nn.Module):

    def __init__(
        self,
        isBatch: bool = False,  # whether 3D keypoints is a mini-batch
        transpose: bool = True,  # whether to transpose the camera matrix
        useFast: bool = False,  # whether to use faster implementation of BPnP
    ):
        super(BPnP, self).__init__()

        self.BPnP_func = (
            BPnPFunction
            if not isBatch
            else BPnPFunction_fast if useFast else BPnPFunction_m3d
        )
        self.transpose = transpose

    def forward(
        self,
        keypoints2d: torch.Tensor,  # [B, K, 2]
        intrinsics: torch.Tensor,  # [3, 3]
        keypoints3d: torch.Tensor,  # [K, 3] or # [B,K,3]
    ) -> torch.Tensor:  # [B, 6]

        intrinsics = intrinsics[0, :, :] if len(intrinsics.shape) == 3 else intrinsics
        intrinsics = intrinsics.t() if self.transpose else intrinsics
        P_6d = self.BPnP_func.apply(keypoints2d, keypoints3d, intrinsics)

        return P_6d
