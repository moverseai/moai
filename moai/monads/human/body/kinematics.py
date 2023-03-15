import torch


class FK(torch.nn.Module):
    def __init__(
        self,
        parents: torch.Tensor,
    ):
        """
        Forward Kinematics
        returns the global poses of the joints
        local_oris: (batch_size, n_joints, 3, 3)
        parents: (n_joints, )
        output: (batch_size, n_joints, 3, 3)
        """
        super(FK, self).__init__()
        self.parents = parents

    def forward(self, local_poses: torch.Tensor) -> torch.Tensor:
        n_joints = len(self.parents)
        global_oris = torch.zeros_like(local_poses)
        for j in range(n_joints):
            if self.parents[j] < 0:  # root rotation
                global_oris[..., j, :, :] = local_poses[..., j, :, :]
            else:
                parent_rot = global_oris[..., self.parents[j], :, :]
                local_rot = local_poses[..., j, :, :]
                global_oris[..., j, :, :] = torch.matmul(parent_rot, local_rot)
        return global_oris
