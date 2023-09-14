import torch
import typing


class DecomposeMatrix(torch.nn.Module):
    """
    Derive Rotation and Translation from a 4x4 matrix.

    Args:
        order (str): The order of the matrix. Either "row" or "column".
    """

    def __init__(
        self,
        order: str = "row",
    ) -> None:
        super().__init__()

        self.order = order

    def forward(
        self,
        matrix: torch.Tensor,
    ) -> typing.Mapping[str, torch.Tensor]:
        if self.order == "row":
            # Extract the translation vector from the bottom row
            t = matrix[:, 3, :3] if len(matrix.shape) == 3 else matrix[:, :, 3, :3]
            # Extract the rotation matrix from the upper-left 3x3 submatrix
            R = matrix[..., :3, :3]
        elif self.order == "column":
            # Extract the rotation matrix from the upper-left 3x3 submatrix
            R = matrix[..., :3, :3] #.transpose(-1, -2)
            # Extract the translation vector from the right-most column
            t = matrix[..., :3, 3]
        else:
            raise ValueError(f"Unknown order: {self.order}")

        return {
            "rotation": R,
            "translation": t,
        }


if __name__ == "__main__":
    # Example batch of 4x4 transformation matrices
    T_batch_column = torch.Tensor(
        [
            [[0.866, -0.5, 0, 1], [0.5, 0.866, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
            [[1, 0, 0, 4], [0, 1, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]],
        ]
    )

    T_batch_row = torch.Tensor(
        [
            [[0.707, -0.707, 0, 0], [0.707, 0.707, 0, 0], [0, 0, 1, 0], [1, 2, 3, 1]],
            [[-0.707, 0.707, 0, 0], [-0.707, -0.707, 0, 0], [0, 0, 1, 0], [1, 2, 3, 1]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [4, 5, 6, 1]],
        ]
    )
    decompose_row = DecomposeMatrix(order="row")

    decompose_column = DecomposeMatrix(order="column")

    print(decompose_row(T_batch_row))
    print(decompose_column(T_batch_column))
