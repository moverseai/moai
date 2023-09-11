import torch

import typing


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


class ForwardKinematics(torch.nn.Module):
    def __init__(
        self,
        parents: torch.Tensor,
    ) -> None:
        super().__init__()
        """
        Forward kinematic process. 
        In forward kinematics, 
        you start with the joint rotations 
        and propagate them from parent joints to child joints to 
        determine the position and orientation of the 
        end effector or joints in global space.

        Parameters
        ----------
        rot_mats : torch.tensor BxNx3x3
            Tensor of rotation matrices
        joints : torch.tensor BxNx3
            Locations of joints
        parents : torch.tensor BxN
            The kinematic tree of each object

        Returns
        -------
        posed_joints : torch.tensor BxNx3
            The locations of the joints after applying the pose rotations
        rel_transforms : torch.tensor BxNx4x4
            The relative (with respect to the root joint) rigid transformations
            for all the joints
        """
        self.parents = parents

    @staticmethod
    def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
        """
        # No padding left or right, only add an extra row
        return torch.cat(
            [
                torch.nn.functional.pad(R, [0, 0, 0, 1]),
                torch.nn.functional.pad(t, [0, 0, 0, 1], value=1),
            ],
            dim=2,
        )

    def forward(
        self, rotation_matrices: torch.Tensor, joints: torch.Tensor  # BxNx3x3  # BxNx3
    ) -> typing.Dict[str, torch.Tensor]:
        joints = torch.unsqueeze(joints, dim=-1)

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, self.parents[1:]]

        transforms_mat = self.transform_mat(
            rotation_matrices.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1).repeat(
                int(rotation_matrices.shape[0] / rel_joints.shape[0]), 1, 1
            ),
        ).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, len(self.parents)):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(
                transform_chain[self.parents[i]], transforms_mat[:, i]
            )
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        # The last column of the transformations contains the posed joints
        posed_joints = transforms[:, :, :3, 3]

        joints_homogen = torch.nn.functional.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - torch.nn.functional.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
        )

        return {
            "posed_joints": posed_joints,
            "rel_transforms": rel_transforms,
        }
