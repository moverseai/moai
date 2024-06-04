import json
import logging
import os
import typing

import numpy as np
import torch
import trimesh

from moai.data.datasets.common.image2d import load_color_image
from moai.utils.arguments import assert_path
from moai.utils.os import glob

log = logging.getLogger(__name__)

__all__ = ["HBW"]


class HBW(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "test",
        single_person: bool = True,
    ) -> None:
        super().__init__()
        assert_path(log, "HBW root path", root)
        self.root = root
        self.split = split
        self.single_person = single_person
        self.images = glob(os.path.join(root, "images", split, "**", "**", "*.png"))
        log.info(f"Loaded {len(self)} items from HBW.")

    def __len__(self) -> int:
        return len(self.images)

    def load_keypoints(self, filename: str) -> typing.Dict[str, torch.Tensor]:
        with open(filename) as keypoint_file:
            data = json.load(keypoint_file)
        persons = []
        for person in data["people"]:
            persons.append({})
            body = np.array(person["pose_keypoints_2d"], dtype=np.float32)
            body = torch.from_numpy(body.reshape([-1, 3]))
            persons[-1]["body"] = {"keypoints": body[:, :2], "confidence": body[:, 2:3]}
            persons[-1]["all"] = {"keypoints": body[:, :2], "confidence": body[:, 2:3]}
            left_hand = np.array(
                person["hand_left_keypoints_2d"], dtype=np.float32
            ).reshape([-1, 3])
            right_hand = np.array(
                person["hand_right_keypoints_2d"], dtype=np.float32
            ).reshape([-1, 3])
            # body = np.concatenate([body, left_hand, right_hand], axis=0)
            left_hand = torch.from_numpy(left_hand)
            right_hand = torch.from_numpy(right_hand)
            persons[-1]["hands"] = {
                "left": {
                    "keypoints": left_hand[:, :2],
                    "confidence": left_hand[:, 2:3],
                },
                "right": {
                    "keypoints": right_hand[:, :2],
                    "confidence": right_hand[:, 2:3],
                },
            }
            persons[-1]["all"]["keypoints"] = torch.cat(
                [
                    persons[-1]["all"]["keypoints"],
                    persons[-1]["hands"]["left"]["keypoints"],
                    persons[-1]["hands"]["right"]["keypoints"],
                ],
                dim=0,
            )
            persons[-1]["all"]["confidence"] = torch.cat(
                [
                    persons[-1]["all"]["confidence"],
                    persons[-1]["hands"]["left"]["confidence"],
                    persons[-1]["hands"]["right"]["confidence"],
                ],
                dim=0,
            )
            face = np.array(person["face_keypoints_2d"], dtype=np.float32).reshape(
                [-1, 3]
            )[17 : 17 + 51, :]
            face = torch.from_numpy(face)
            persons[-1]["face"] = {"keypoints": face[:, :2], "confidence": face[:, 2:3]}
            # contour_keyps = np.array([], dtype=body.dtype).reshape(0, 3)
            persons[-1]["all"]["keypoints"] = torch.cat(
                [
                    persons[-1]["all"]["keypoints"],
                    persons[-1]["face"]["keypoints"],
                ],
                dim=0,
            )
            persons[-1]["all"]["confidence"] = torch.cat(
                [
                    persons[-1]["all"]["confidence"],
                    persons[-1]["face"]["confidence"],
                ],
                dim=0,
            )
            contour_keyps = np.array(
                person["face_keypoints_2d"], dtype=np.float32
            ).reshape([-1, 3])[:17, :]
            contour_keyps = torch.from_numpy(contour_keyps)
            persons[-1]["face_contour"] = {
                "keypoints": contour_keyps[:, :2],
                "confidence": contour_keyps[:, 2:3],
            }
            persons[-1]["all"]["keypoints"] = torch.cat(
                [
                    persons[-1]["all"]["keypoints"],
                    persons[-1]["face_contour"]["keypoints"],
                ],
                dim=0,
            )
            persons[-1]["all"]["confidence"] = torch.cat(
                [
                    persons[-1]["all"]["confidence"],
                    persons[-1]["face_contour"]["confidence"],
                ],
                dim=0,
            )
            gender = (
                person.get("gender_gt", None)
                or person.get("gender", None)
                or person.get("gender_pd", None)
            )
            if gender is not None:
                persons[-1]["gender"]
        if not self.single_person:
            return persons
        else:

            def _get_area(person: typing.Dict[str, torch.Tensor]) -> float:
                keypoints = person["body"]["keypoints"]
                confidence = person["body"]["confidence"]
                # TODO: update selection with threshold based discarding?
                min_x = keypoints[..., 0].min()
                min_y = keypoints[..., 1].min()
                max_x = keypoints[..., 0].max()
                max_y = keypoints[..., 1].max()
                return (max_x - min_x) * (max_y - min_y) * confidence.sum()

            person = max(persons, key=_get_area)
            return person

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        image_filename = self.images[index]
        kpts_filename = image_filename.replace("images", "keypoints").replace(
            ".png", ".json"
        )
        data = {
            "image": load_color_image(image_filename),
            "keypoints": self.load_keypoints(kpts_filename),
            "__moai__": {
                "dataset": type(self).__name__,
            },
        }
        if self.split == "val":
            id = os.path.basename(
                os.path.dirname(os.path.dirname(image_filename))
            ).split("_")[0]
            mesh = trimesh.load(
                os.path.join(self.root, "smplx", "val", f"{id}.obj"), process=False
            )
            data["smplx"] = {
                "mesh": {
                    "vertices": torch.from_numpy(mesh.vertices).float(),
                    "faces": torch.from_numpy(mesh.faces).long(),
                }
            }
        return data
