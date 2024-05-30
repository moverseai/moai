import logging
import typing
import functools
import torch
import numpy as np
import math
import toolz
import colour

from moai.utils.arguments import assert_numeric

log = logging.getLogger(__name__)

from collections.abc import Callable

try:
    import rerun as rr
except:
    log.error(f"Please `pip install rerun-sdk` to use ReRun visualisation module.")


class Points3d(Callable):
    def __init__(
        self,
        name: str,  # Path to the points in the space hierarchy,
        points: typing.Union[str, typing.Sequence[str]],
        color: typing.Union[str, typing.Sequence[str]],
        path_to_hierarchy: typing.Sequence[str],
        batch_percentage: float = 1.0,
        skeleton: typing.Sequence[int]=None,
    ):
        # rr.init(name)
        # rr.spawn()
        self.name = name
        self.keys = [points] if isinstance(points, str) else list(points)
        self.path_to_hierarchy = path_to_hierarchy
        self.colors = (
            [colour.Color(color)]
            if isinstance(color, str)
            else list(colour.Color(c) for c in color)
        )
        self.batch_percentage = batch_percentage
        assert_numeric(log, 'batch percentage', batch_percentage, 0.0, 1.0)
        # self.images = [image] if isinstance(image, str) else list(image)
        self.access = lambda td, k: toolz.get_in(k.split('.'), td)
        self.step = 0
        self.kintree = skeleton

    @staticmethod  # TODO: refactor these into a common module
    def _passthrough(
        tensors: typing.Dict[str, torch.Tensor],
        key: str,
        count: int,
    ) -> torch.Tensor:
        return tensors[key][:count] if key in tensors else None

    def __call__(
        self, tensors: typing.Dict[str, torch.Tensor], step: typing.Optional[int] = None
    ) -> None:
        for k, c, path in zip(self.keys, self.colors, self.path_to_hierarchy):
            rr.set_time_sequence("frame_nr", self.step)
            rr.log_points(
                path, # where to put the points in the hierarchy
                self.access(tensors, k).detach().cpu().numpy()[0],
                colors=c.get_rgb(),
            )
            # log skeleton also
            # DEBUG; will be removed
            if 'joints' in k:
                kpts = self.access(tensors, k)
                edges = []
                for nk, (i, j) in enumerate(self.kintree):
                    edges.append(kpts[0][i].cpu().numpy())
                    edges.append(kpts[0][j].cpu().numpy())
                if 'predicted' in path:
                    rr.log_line_segments('world/predicted/pose', edges, color = [0.1,0.1,0.1])
                    # add also 2d pose
                    pose = tensors['joints3d_predicted_filtered_d_cam_0']
                    kpts2d = pose[:,[1,0],:].permute(0,2,1)
                    edges = []
                    kpts2d[:,:,[1]] = 1280 - kpts2d[:,:,[1]] 
                    for nk, (i, j) in enumerate(self.kintree):
                        edges.append(kpts2d[0][i].cpu().numpy())
                        edges.append(kpts2d[0][j].cpu().numpy())
                    rr.log_points('rgb_cam_0/predicted/pose/points', kpts2d[0])
                    rr.log_line_segments('rgb_cam_0/predicted/pose/skeleton', edges, color = [1.0,0.1,0.1])
                elif 'fitting' in path:
                    if 'temporal' in path:
                        rr.log_line_segments('world/temporal_fitting/pose', edges, color = [0.1,0.1,0.1])
                        # add also 2d pose
                        pose = tensors['body_joints3d_t_cam_0']
                        kpts2d = pose[:,[1,0],:].permute(0,2,1)
                        edges = []
                        kpts2d[:,:,[1]] = 1280 - kpts2d[:,:,[1]] 
                        for nk, (i, j) in enumerate(self.kintree):
                            edges.append(kpts2d[0][i].cpu().numpy())
                            edges.append(kpts2d[0][j].cpu().numpy())
                        rr.log_points('rgb_cam_0/temporal_fitting/pose/points', kpts2d[0])
                        rr.log_line_segments('rgb_cam_0/temporal_fitting/pose/skeleton', edges, color = [0.1,0.1,1.0])
                    else:
                        rr.log_line_segments('world/fitting/pose', edges, color = [0.1,0.1,0.1])
                        # add also 2d pose
                        pose = tensors['body_joints3d_cam_0']
                        kpts2d = pose[:,[1,0],:].permute(0,2,1)
                        edges = []
                        kpts2d[:,:,[1]] = 1280 - kpts2d[:,:,[1]] 
                        for nk, (i, j) in enumerate(self.kintree):
                            edges.append(kpts2d[0][i].cpu().numpy())
                            edges.append(kpts2d[0][j].cpu().numpy())
                        rr.log_points('rgb_cam_0/fitting/pose/points', kpts2d[0])
                        rr.log_line_segments('rgb_cam_0/fitting/pose/skeleton', edges, color = [0.1,0.1,0.1])

                            
            # DEBUG


            
        self.step += 1
