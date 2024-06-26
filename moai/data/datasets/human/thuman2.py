import glob
import logging
import os
import pickle
import typing

import numpy as np
import smplx
import toolz
import torch
import tqdm
from scipy.spatial.transform import Rotation as R

from moai.utils.arguments import assert_path

log = logging.getLogger(__name__)

__all__ = ["THuman2"]


class THuman2(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        smplx_root: str,
        reconstruct: bool = False,
    ) -> None:
        super().__init__()
        assert_path(log, "THuman2.0 data root path", data_root)
        assert_path(log, "SMPLX data path", smplx_root)
        self.data = []
        self.reconstruct = reconstruct
        for sample in tqdm.tqdm(
            glob.glob(os.path.join(data_root, "smplx", "**", "*.pkl")),
            desc="Loading THuman2.0 data",
        ):
            with open(sample, "rb") as f:
                data = pickle.load(f)
            self.data.append(
                toolz.valmap(lambda a: torch.from_numpy(a.squeeze()).float(), data)
            )
        if self.reconstruct:
            self.body = smplx.create(
                model_path=smplx_root,
                model_type="smplx",
                num_expression_coeffs=10,
                create_expression=True,
                create_jaw_pose=False,
                create_leye_pose=False,
                create_reye_pose=False,
                use_face_contour=False,
                batch_size=1,
                age="adult",
                create_left_hand_pose=False,
                create_right_hand_pose=False,
                use_pca=True,
                num_pca_comps=12,
                gender="neutral",
                flat_hand_mean=False,
                create_betas=False,
                num_betas=10,
                create_global_orient=False,
                create_body_pose=False,
                create_transl=False,
            )
            self.body.requires_grad_(False)
        log.info(f"Loaded {len(self)} THuman2.0 samples.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        item = self.data[index]
        # fix global rotation
        r = R.from_euler("xz", [90, -90], degrees=True)
        new_global_orientation = R.from_matrix(
            (r.as_matrix() @ R.from_rotvec(item["global_orient"]).as_matrix())
        ).as_rotvec()
        out = {
            "smplx": {
                "gender": 0,
                "scale": item["scale"],
                "params": {
                    "betas": item["betas"],
                    "expression": item["expression"],
                    "body_pose": item["body_pose"],
                    "transl": item["translation"],
                    "global_orient": torch.from_numpy(new_global_orientation).float(),
                    "left_hand_pose": item["left_hand_pose"],
                    "right_hand_pose": item["right_hand_pose"],
                    "jaw_pose": item["jaw_pose"],
                    "left_eye_pose": item["leye_pose"],
                    "right_eye_pose": item["reye_pose"],
                },
            },
            "__moai__": {
                "dataset": type(self).__name__,
            },
        }
        if self.reconstruct:
            with torch.no_grad():
                body = self.body.forward(
                    global_orient=torch.from_numpy(new_global_orientation).float()[
                        np.newaxis, ...
                    ],
                    betas=out["smplx"]["params"]["betas"][np.newaxis, ...],
                    body_pose=out["smplx"]["params"]["body_pose"][np.newaxis, ...],
                    left_hand_pose=out["smplx"]["params"]["left_hand_pose"][
                        np.newaxis, ...
                    ],
                    right_hand_pose=out["smplx"]["params"]["right_hand_pose"][
                        np.newaxis, ...
                    ],
                    jaw_pose=out["smplx"]["params"]["jaw_pose"][np.newaxis, ...],
                    leye_pose=out["smplx"]["params"]["left_eye_pose"][np.newaxis, ...],
                    reye_pose=out["smplx"]["params"]["right_eye_pose"][np.newaxis, ...],
                    transl=out["smplx"]["params"]["transl"][np.newaxis, ...],
                    expression=out["smplx"]["params"]["expression"][np.newaxis, ...],
                )
                out["smplx"].update(
                    {
                        "mesh": {
                            "vertices": body.vertices[0],
                            "faces": self.body.faces_tensor,
                        },
                        "joints": body.joints[0, :55, ...],
                    }
                )
        return out
