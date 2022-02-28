from moai.visualization.visdom.base import Base
from moai.utils.arguments import ensure_string_list
from moai.utils.iterators import pairwise

import torch
import visdom
import functools
import typing
import logging
import numpy as np
import cv2
import colour
import math
import toolz
from PIL import Image

log = logging.getLogger(__name__)

__all__ = ["Pose2d"]

class Pose2d(Base):
    def __init__(self,
        images:         typing.Union[str, typing.Sequence[str]],
        poses:          typing.Union[str, typing.Sequence[str]],
        gt:             typing.Union[str, typing.Sequence[str]],
        pred:           typing.Union[str, typing.Sequence[str]],
        gt_masks:       typing.Union[str, typing.Sequence[str]],
        pred_masks:     typing.Union[str, typing.Sequence[str]],
        pose_structure: typing.Union[str, typing.Sequence[str]],
        coords:         typing.Union[str, typing.Sequence[str]],
        color_gt:       typing.Union[str, typing.Sequence[str]],
        color_pred:     typing.Union[str, typing.Sequence[str]],
        name:           str="default",
        ip:             str="http://localhost",
        port:           int=8097,
        reverse_coords: bool=False,
        rotate_image:   bool=False,
        transparency:   float=0.4,
        scale:          float=1.0,
        use_mask:       bool=True,
    ):
        super(Pose2d, self).__init__(name, ip, port)
        self.images = ensure_string_list(images)
        self.poses = ensure_string_list(poses)
        self.gt = ensure_string_list(gt)
        self.pred = ensure_string_list(pred)
        self.gt_masks = ensure_string_list(gt_masks)
        self.pred_masks = ensure_string_list(pred_masks)
        self.pose_structure = ensure_string_list(pose_structure)
        self.coords = ensure_string_list(coords)
        self.color_gt = list(map(colour.web2rgb, ensure_string_list(color_gt)))
        self.color_pred = list(map(colour.web2rgb, ensure_string_list(color_pred)))
        self.reverse = reverse_coords
        self.rotate = rotate_image
        self.transparency = transparency
        self.scale = scale
        self.use_mask = use_mask
        self.viz_pose = {
            'human_pose2d': functools.partial(self.__draw_human_pose2d, 
                self.visualizer, marker=cv2.MARKER_DIAMOND, 
                rotate=self.rotate, transparency=self.transparency,
                scale=self.scale
            ),
        }
        self.xforms = { #TODO: extract these into a common module
            'ndc': lambda coord, img: torch.addcmul(
                torch.scalar_tensor(0.5).to(coord), coord, torch.scalar_tensor(0.5).to(coord)
            ) * torch.Tensor([*img.shape[2:]]).to(coord).expand_as(coord),
            'coord': lambda coord, img: coord,
            'norm': lambda coord, img: coord * torch.Tensor([*img.shape[2:]]).to(coord).expand_as(coord),
        }
        self.access = lambda td, k: toolz.get_in(k.split('.'), td)

    @property
    def name(self) -> str:
        return self.env_name
        
    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        for img, poses, gt, pred, gt_masks, pred_masks, pose_struct, gt_c, pred_c, coord in zip(
            self.images, self.poses, self.gt, self.pred, self.gt_masks, self.pred_masks,  [self.pose_structure, ],
            self.color_gt, self.color_pred, self.coords
        ):
            gt_coord = self.access(tensors, gt).detach()
            pred_coord = self.access(tensors, pred).detach()
            gt_masks = self.access(tensors, gt_masks)
            pred_masks = self.access(tensors, pred_masks)
            if self.reverse:
                gt_coord = gt_coord.flip(-1)
                pred_coord = pred_coord.flip(-1)
            image = self.access(tensors, img).detach()
            self.viz_pose[poses](
                image,
                self.xforms[coord](gt_coord, image),
                self.xforms[coord](pred_coord, image),
                gt_masks.detach() if self.use_mask else torch.ones_like(gt_coord[..., 0:1]),
                pred_masks.detach() if self.use_mask else torch.ones_like(gt_coord[..., 0:1]),
                pose_struct,
                np.uint8(np.array(list(gt_c)) * 255),
                np.uint8(np.array(list(pred_c)) * 255),
                # np.uint8(np.array(list(reversed(gt_c))) * 255),
                # np.uint8(np.array(list(reversed(pred_c))) * 255),
                coord, img, img, self.name
            )    
    
    @staticmethod
    def __draw_human_pose2d(
        visdom:             visdom.Visdom,
        images:             torch.Tensor,
        gt_coordinates:     torch.Tensor,
        pred_coordinates:   torch.Tensor,
        gt_masks:           torch.Tensor,
        pred_masks:         torch.Tensor,
        pose_structure:     typing.List[typing.List[int]],
        gt_color:           typing.List[float],
        pred_color:         typing.List[float],
        coord:              str,
        key:                str,
        win:                str,
        env:                str,
        marker:             int,
        rotate:             bool,
        transparency:       float,
        scale:              float,
    ):
        b, _, h, w = images.shape
        imgs = np.zeros([b, 3, int(scale * h), int(scale * w)], dtype=np.uint8) if not rotate \
            else np.zeros([b, 3, int(scale * w), int(scale * h)], dtype=np.uint8)
        gt_coords = gt_coordinates.cpu().int()
        pred_coords = pred_coordinates.cpu().int()
        gt_coords = torch.flip(gt_coords, dims=[-1])
        pred_coords = torch.flip(pred_coords, dims=[-1])
        gt_coords = gt_coords.numpy()
        pred_coords = pred_coords.numpy()
        diagonal = torch.norm(torch.Tensor([*imgs.shape[2:]]), p=2)
        marker_size = int(0.015 * diagonal) #TODO: extract percentage param to config?
        line_size = int(0.005 * diagonal) #TODO: extract percentage param to config?
        for i in range(imgs.shape[0]):
            img = images[i, ...].cpu().numpy().transpose(1, 2, 0) * 255.0
            img = img.copy().astype(np.uint8) if img.shape[2] > 1\
                else cv2.cvtColor(img.copy().astype(np.uint8), cv2.COLOR_GRAY2RGB)
            bg = img.copy()
            for coords, color, masks in zip(
                [gt_coords, pred_coords],
                [gt_color, pred_color],
                [gt_masks, pred_masks]
            ):       
                coord_i = coords[i, ...]
                for kpts_group in pose_structure:
                    for (a, b) in pairwise(kpts_group):
                    # for j in range(len(kpts_group) - 1):                    
                        if torch.sum(masks[i, a]) and torch.sum(masks[i, b]):
                            start_xy = tuple(coord_i[a])
                            end_xy = tuple(coord_i[b])
                            X = (start_xy[0], end_xy[0])
                            Y = (start_xy[1], end_xy[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = line_size
                            # polygon = cv2.ellipse2Poly(
                            #     (int(mX or 0), int(mY or 0)),
                            #     (int(length/2 or 1), int(stickwidth or 1)),
                            #     int(angle), 0, 360, 1
                            # )
                            # cv2.fillConvexPoly(bg, polygon, color.tolist())
                            cv2.line(img, start_xy, end_xy, color.tolist(), thickness=line_size)
                for k, coord in enumerate(coord_i):
                    if torch.sum(masks[i, k]):
                        if marker < 0:
                            cv2.circle(bg, 
                                tuple(coord), 
                                marker_size, color.tolist(), thickness=line_size
                            )
                        else:
                            cv2.drawMarker(bg, 
                                tuple(coord),
                                color.tolist(),
                                marker, marker_size, line_size
                            )

                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
            h, w = img.shape[:2]
            if scale != 1.0:
                img = np.array(Image.fromarray(img).resize(
                    (int(w * scale), int(h * scale)), Image.ANTIALIAS
                ))
            imgs[i, ...] = img.transpose(2, 0, 1) if not rotate\
                else cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE).transpose(2, 0, 1)
        visdom.images(
            imgs,
            # np.flip(imgs, axis=1),
            win=win,
            env=env,
            opts={
                'title': key,
                'caption': key,
                'jpgquality': 50,
            }
        )
