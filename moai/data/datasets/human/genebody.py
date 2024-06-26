import glob
import logging
import os
import typing

import numpy as np
import toolz
import torch
import trimesh

from moai.data.datasets.common.image2d import load_color_image, load_mask_image
from moai.utils.arguments import assert_path

log = logging.getLogger(__name__)

__all__ = ["GeneBody"]


class GeneBody(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "test",
        subjects: typing.Union[typing.List[str], str] = "**",
        cameras: typing.Union[
            typing.List[str], str
        ] = "**",  # NOTE: the default seting of GNR is to use these four source views of GeneBody: ['01', '13', '25', '37']
        # modalities:         typing.List[str]=['image'],
    ) -> None:
        super().__init__()
        assert_path(log, "GenBody root path", root)
        is_all_subjects = isinstance(subjects, str) and (
            "all" == subjects or "**" == subjects
        )
        subjects = (
            os.listdir(os.path.join(root, split)) if is_all_subjects else subjects
        )
        subjects = [subjects] if isinstance(subjects, str) else subjects
        is_all_cameras = isinstance(cameras, str) and (
            "all" == cameras or "**" == cameras
        )
        cameras = (
            os.listdir(os.path.join(root, split, "image"))
            if is_all_cameras
            else cameras
        )
        cameras = [cameras] if isinstance(cameras, str) else cameras
        self.metadata = {
            "root": root,
            "items": [],
            "subjects": {},
            "cameras": np.array(cameras).astype(np.int32),
        }
        for subject in subjects:
            params = glob.glob(os.path.join(root, split, subject, "param", "*.*"))
            cams = np.load(
                os.path.join(root, split, subject, "annots.npy"), allow_pickle=True
            ).item()["cams"]
            extra = "0" if subject == "zhuna" else ""
            for param in params:
                filename = os.path.basename(param)
                self.metadata["items"].append(
                    {
                        "smplx": param,
                        "mesh": param.replace("param", "smpl").replace("npy", "obj"),
                        "images": {
                            c: os.path.join(
                                root,
                                split,
                                subject,
                                "image",
                                c,
                                filename.replace("npy", "jpg"),
                            )
                            for c in cameras
                        },
                        "masks": {
                            c: os.path.join(
                                root,
                                split,
                                subject,
                                "mask",
                                c,
                                "mask" + extra + filename.replace("npy", "png"),
                            )
                            for c in cameras
                        },
                        "cameras": toolz.valmap(
                            lambda d: toolz.valmap(
                                lambda v: torch.from_numpy(v).squeeze().float(), d
                            ),
                            toolz.keyfilter(lambda k: k in cameras, cams),
                        ),
                    }
                )
        log.info(f"Loaded {len(self)} items from GeneBody.")

    def __len__(self) -> int:
        return len(self.metadata["items"])

    def load_smplx(self, filename: str) -> typing.Dict[str, torch.Tensor]:
        data = np.load(filename, allow_pickle=True).item()
        scale = torch.scalar_tensor(data["smplx_scale"]).float()
        # data = toolz.valmap(lambda v: torch.from_numpy(v.squeeze()).float(), data['smplx'])
        data = toolz.valmap(
            lambda v: (
                torch.from_numpy(v.squeeze()).float()
                if type(v) is not torch.Tensor
                else v.squeeze()
            ),
            data["smplx"],
        )
        return {
            "scale": scale,
            "params": toolz.dissoc(data, "keypoints3d"),
            "keypoints3d": data["keypoints3d"],
        }

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        item = self.metadata["items"][index]
        mesh = trimesh.load(item["mesh"], process=False)
        smplx = self.load_smplx(item["smplx"])
        data = {
            "smplx": toolz.merge(
                smplx,
                {
                    "gender": 0,  # neutral
                    "mesh": {
                        "vertices": torch.from_numpy(mesh.vertices).float(),
                        "scaled_vertices": smplx["scale"]
                        * torch.from_numpy(mesh.vertices).float(),
                        "faces": torch.from_numpy(mesh.faces).int(),  # NOTE: or long?
                    },
                    "joints": smplx["keypoints3d"],
                },
            ),
            "images": toolz.valmap(load_color_image, item["images"]),
            "masks": toolz.valmap(load_mask_image, item["masks"]),
            "cameras": item["cameras"],
            "__moai__": {
                "dataset": type(self).__name__,
            },
        }
        return data


if __name__ == "__main__":
    # initiate the dataset
    dataset = GeneBody(
        root="//192.168.1.3/Public/Human Datasets/GeneBody",
        split="train",
        subjects="ahha",
        cameras=["01"],
        # smplx_root="C:\\Users\\giorg\\Documents\\Projects\\markerless-mocap\\third_party\\smplx\\transfer_data\\body_models",
        # reconstruct=True,
    )
    # iterate over the dataset
    for i in range(len(dataset)):
        print(dataset[i])
