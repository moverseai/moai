import typing

import torch

from moai.utils.torch import inverse_sigmoid

__all__ = ["GaussianSplattingParameters"]


class GaussianSplattingParameters(torch.nn.Module):
    def __init__(
        self,
        num_splats: int,
        num_sets: int,
        skip_positions: bool = False,
        scale: typing.Union[float, typing.Sequence[float]] = 1.0,
        opacity: typing.Optional[float] = None,
        sh_degree: int = 0,
    ):
        super().__init__()
        assert num_splats > 0 and num_sets > 0
        self.has_positions = not skip_positions
        self.sh_degree = sh_degree
        if self.has_positions:
            self.register_parameter(
                "position", torch.nn.Parameter(torch.zeros(num_sets, num_splats, 3))
            )
        self.register_parameter(
            "rotation",
            torch.nn.Parameter(
                torch.cat(
                    [
                        torch.ones(num_sets, num_splats, 1),
                        torch.zeros(num_sets, num_splats, 3),
                    ],
                    dim=-1,
                )
            ),
        )
        # self.register_parameter(
        #     "spherical_harmonics",
        #     torch.nn.Parameter(
        #         torch.cat(
        #             [
        #                 torch.zeros(num_sets, num_splats, 1, 3),
        #                 torch.zeros(num_sets, num_splats, (sh_degree + 1) ** 2 - 1, 3),
        #             ],
        #             dim=-2,
        #         )
        #     ),
        # )
        self.register_parameter(
            "spherical_harmonics_dc",
            torch.nn.Parameter(
                torch.zeros(num_sets, num_splats, 1, 3),
            ),
        )
        for l in range(sh_degree):
            self.register_parameter(
                f"spherical_harmonics_d{l+1}",
                torch.nn.Parameter(
                    torch.zeros(num_sets, num_splats, 2 * (l + 1) + 1, 3),
                ),
            )
        scaling = (
            torch.tensor([*scale])
            if not isinstance(scale, float)
            else torch.scalar_tensor(scale)
        )
        assert scaling.abs().sum() > 1e-7
        self.register_parameter(
            "scaling",
            torch.nn.Parameter(
                torch.log(torch.ones(num_sets, num_splats, 3) * scaling)
            ),
        )
        assert opacity is None or (opacity > 0 and opacity < 1)
        self.register_parameter(
            "opacity",
            torch.nn.Parameter(
                inverse_sigmoid(torch.full((num_sets, num_splats, 1), opacity or 0.99))
            ),
        )

    def forward(self, tensor: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
        params = {
            "rotation": torch.nn.functional.normalize(self.rotation, p=2, dim=-1),
            "opacity": torch.sigmoid(self.opacity),
            "spherical_harmonics_dc": self.spherical_harmonics_dc,  # NOTE: relu?
            "scaling": torch.exp(self.scaling),
        }
        for l in range(self.sh_degree):
            key = f"spherical_harmonics_d{l + 1}"
            params[key] = getattr(self, key)
        if self.has_positions:
            params["position"] = self.position
        return params
