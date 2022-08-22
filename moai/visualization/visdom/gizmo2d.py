from moai.visualization.visdom.base import Base
from moai.utils.arguments import ensure_string_list
from moai.monads.execution.cascade import _create_accessor

import torch
import visdom
import functools
import typing
import logging
import numpy as np
import cv2
import colour

log = logging.getLogger(__name__)

__all__ = ["Gizmo2d"]

class Gizmo2d(Base):
    def __init__(self,
        images:         typing.Union[str, typing.Sequence[str]],
        gizmos:         typing.Union[str, typing.Sequence[str]],
        gt:             typing.Union[str, typing.Sequence[str]],
        pred:           typing.Union[str, typing.Sequence[str]],
        coords:         typing.Union[str, typing.Sequence[str]],
        color_gt:       typing.Union[str, typing.Sequence[str]],
        color_pred:     typing.Union[str, typing.Sequence[str]],
        name:           str="default",
        ip:             str="http://localhost",
        port:           int=8097,
        reverse_coords: bool=False,
    ):
        super(Gizmo2d, self).__init__(name, ip, port)
        self.images = ensure_string_list(images)
        self.gizmos = ensure_string_list(gizmos)
        self.gt = ensure_string_list(gt)
        self.gt = [_create_accessor(k) for k in self.gt]
        self.pred = ensure_string_list(pred)
        self.pred = [_create_accessor(k) for k in self.pred]
        self.color_gt = list(map(colour.web2rgb, ensure_string_list(color_gt)))
        self.color_pred = list(map(colour.web2rgb, ensure_string_list(color_pred)))
        self.coords = ensure_string_list(coords)
        self.reverse = reverse_coords
        self.gizmo_render = {
            'marker_circle': functools.partial(self.__draw_markers, 
                self.visualizer, marker=-1),
            'marker_diamond': functools.partial(self.__draw_markers, 
                self.visualizer, marker=cv2.MARKER_DIAMOND),
            'marker_star': functools.partial(self.__draw_markers, 
                self.visualizer, marker=cv2.MARKER_STAR),
            'marker_cross': functools.partial(self.__draw_markers, 
                self.visualizer, marker=cv2.MARKER_CROSS),
            'marker_square': functools.partial(self.__draw_markers, 
                self.visualizer, marker=cv2.MARKER_SQUARE),
            'bbox2d': functools.partial(self.__draw_2dbox, self.visualizer),
            'bbox3d': functools.partial(self.__draw_3dbox, self.visualizer),
            'axes': functools.partial(self.__draw_axes, self.visualizer),
            #TODO: axes to only receive a scale parameter and have the axis points hardcoded here
        }
        self.xforms = { #TODO: extract these into a common module
            'ndc': lambda coord, img: torch.addcmul(
                torch.scalar_tensor(0.5).to(coord), coord, torch.scalar_tensor(0.5).to(coord)
            ) * torch.Tensor([*img.shape[2:]]).to(coord).expand_as(coord),
            'coord': lambda coord, img: coord,
            'norm': lambda coord, img: coord * torch.Tensor([*img.shape[2:]]).to(coord).expand_as(coord),
        }

    @property
    def name(self) -> str:
        return self.env_name
        
    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        for img, gzm, gt, pred, gt_c, pred_c, coord in zip(
            self.images, self.gizmos, self.gt, self.pred,
            self.color_gt, self.color_pred, self.coords
        ):
            #gt_coord = tensors[gt].detach()
            #pred_coord = tensors[pred].detach()
            gt_coord = gt(tensors).detach()
            pred_coord = pred(tensors).detach()
            if self.reverse:
                gt_coord = gt_coord.flip(-1)
                pred_coord = pred_coord.flip(-1)
            image = tensors[img].detach()
            self.gizmo_render[gzm](
                image,
                self.xforms[coord](gt_coord, image),
                self.xforms[coord](pred_coord, image),
                np.uint8(np.array(list((gt_c))) * 255),
                np.uint8(np.array(list((pred_c))) * 255),             
                coord, img, img, self.name
            )
    
    @staticmethod
    def __draw_2dbox(
        visdom:             visdom.Visdom,
        images:             torch.Tensor,
        gt_coordinates:     torch.Tensor,
        pred_coordinates:   torch.Tensor,
        gt_color:           typing.List[float],
        pred_color:         typing.List[float],
        coord:              str,
        key:                str,
        win:                str,
        env:                str
    ) -> None:
        b, c, h, w = images.size()
        imgs = np.zeros(images.shape, dtype=np.uint8)
        gt_coords = gt_coordinates.detach().cpu()
        pred_coords = pred_coordinates.detach().cpu()
        gt_coords = gt_coords.numpy()
        pred_coords = pred_coords.numpy()
        diagonal = torch.norm(torch.Tensor([*imgs.shape[2:]]), p=2)
        line_size = int(0.005 * diagonal) #TODO: extract percentage param to config?

        for i in range(imgs.shape[0]):
            img = images[i, ...].detach().cpu().numpy().transpose(1, 2, 0) * 255.0
            img = img.copy().astype(np.uint8)  
            for coords, color in zip(
                [gt_coords, pred_coords],
                [gt_color, pred_color]
            ):
                coord_i = coords[i, ...]
                pt1_x , pt1_y , w , h = coord_i
                pt2_x= pt1_x + w   # bottom right
                pt2_y = pt1_y + h   # bottom right
                cv2.rectangle(img,
                    (pt1_x, pt1_y),
                    (pt2_x, pt2_y),
                    color.tolist(),
                    line_size
                )
                imgs[i, ...] = img.transpose(2, 0, 1)
                del coord_i


        visdom.images(
            np.flip(imgs, axis=1),
            win=win,
            env=env,
            opts={
                'title': key,
                'caption': key,
                'jpgquality': 50,
            }
        )
    
    @staticmethod
    def __draw_3dbox(
        visdom:             visdom.Visdom,
        images:             torch.Tensor,
        gt_coordinates:     torch.Tensor,
        pred_coordinates:   torch.Tensor,
        gt_color:           typing.List[float],
        pred_color:         typing.List[float],
        coord:              str,
        key:                str,
        win:                str,
        env:                str
    ) -> None:
        b , c , h , w = images.size()
        imgs = np.zeros(images.shape, dtype=np.uint8)
        gt_coords = gt_coordinates.cpu()
        pred_coords = pred_coordinates.cpu()
        gt_coords = torch.flip(gt_coords, dims=[-1])
        pred_coords = torch.flip(pred_coords, dims=[-1])
        gt_coords = gt_coords.numpy()
        pred_coords = pred_coords.numpy()
        diagonal = torch.norm(torch.Tensor([*imgs.shape[2:]]), p=2)
        line_size = int(0.005 * diagonal) #TODO: extract percentage param to config?
        
        for i in range(imgs.shape[0]):
            img = images[i, ...].detach().cpu().numpy().transpose(1, 2, 0) * 255.0
            img = img.copy().astype(np.uint8)  
            for coords, color in zip(
                [gt_coords, pred_coords],
                [gt_color, pred_color]
            ):
                coord_i = coords[i, ...]
                for k, key_ in enumerate(coord_i):
                    if (k + 1) % 2 == 0:
                        cv2.line(img, (int(point_1_x), int(point_1_y)), (int(coord_i[k][0]), int(coord_i[k][1])), color.tolist(), line_size)
                    else:
                        point_1_x = coord_i[k][0]
                        point_1_y = coord_i[k][1]
                for k, key_ in enumerate(coord_i):
                    if k == 0 or k == 1 or k ==4 or k == 5:
                        point_2_x = coord_i[k+2][0]
                        point_2_y = coord_i[k+2][1]
                        cv2.line(img, (int(point_2_x), int(point_2_y)), (int(coord_i[k][0]), int(coord_i[k][1])), color.tolist(), line_size)
                for k, key_ in enumerate(coord_i):
                    if k == 0 or k == 1 or k == 2 or k == 3:
                        point_2_x = coord_i[k+4][0]
                        point_2_y = coord_i[k+4][1]
                        cv2.line(img, (int(point_2_x), int(point_2_y)), (int(coord_i[k][0]), int(coord_i[k][1])), color.tolist(), line_size)
                imgs[i, ...] = img.transpose(2, 0, 1)
                del coord_i

        visdom.images(
            np.flip(imgs, axis=1),
            win=win,
            env=env,
            opts={
                'title': key,
                'caption': key,
                'jpgquality': 50,
            }
        )

    @staticmethod
    def __draw_axes(
        visdom:             visdom.Visdom,
        images:             torch.Tensor,
        gt_axes:            torch.Tensor,
        pred_axes:          torch.Tensor,
        gt_color:           typing.List[float],
        pred_color:         typing.List[float],
        coord:              str,
        key:                str,
        win:                str,
        env:                str
    ) -> None:
        b , c , h , w = images.size()
        imgs = np.zeros(images.shape, dtype=np.uint8)
        gt_axes = gt_axes.cpu()
        pred_axes = pred_axes.cpu()
        gt_axes = torch.flip(gt_axes, dims=[-1])
        pred_axes = torch.flip(pred_axes, dims=[-1]) 
        gt_axes = gt_axes.numpy()
        pred_axes = pred_axes.numpy()
        diagonal = torch.norm(torch.Tensor([*imgs.shape[2:]]), p=2)
        line_size = int(0.005 * diagonal)
        for i in range(imgs.shape[0]):
            img = images[i, ...].detach().cpu().numpy().transpose(1, 2, 0) * 255.0
            img = img.copy().astype(np.uint8)
            for j , (coords, color) in enumerate(zip(
                [gt_axes, pred_axes],
                [gt_color, pred_color]
            )):
                coord_i = np.int32(coords[i, ...])
                #draw lines
                if j == 1:
                    alpha = 0.4
                    overlay = img.copy()
                    cv2.arrowedLine(overlay,tuple(coord_i[0]),tuple(coord_i[1]), (0,0,255), line_size)
                    cv2.arrowedLine(overlay,tuple(coord_i[0]),tuple(coord_i[2]), (255,0,0), line_size)
                    cv2.arrowedLine(overlay,tuple(coord_i[0]),tuple(coord_i[3]), (0,255,0), line_size)
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha,
		                0, img)
                else:
                    cv2.arrowedLine(img,tuple(coord_i[0]),tuple(coord_i[1]), (0,0,255), line_size)
                    cv2.arrowedLine(img,tuple(coord_i[0]),tuple(coord_i[2]), (255,0,0), line_size)
                    cv2.arrowedLine(img,tuple(coord_i[0]),tuple(coord_i[3]), (0,255,0), line_size)
                
                imgs[i, ...] = img.transpose(2, 0, 1)
                del coord_i
        
        
        visdom.images(
            np.flip(imgs, axis=1),
            win=win,
            env=env,
            opts={
                'title': key,
                'caption': key,
                'jpgquality': 50,
            }
        ) 
    
    @staticmethod
    def __draw_markers(
        visdom:             visdom.Visdom,
        images:             torch.Tensor,
        gt_coordinates:     torch.Tensor,
        pred_coordinates:   torch.Tensor,
        gt_color:           typing.List[float],
        pred_color:         typing.List[float],
        coord:              str,
        key:                str,
        win:                str,
        env:                str,
        marker:             int,
    ):
        imgs = np.zeros([images.shape[0], 3, images.shape[2], images.shape[3]], dtype=np.uint8)
        gt_coords = gt_coordinates.cpu()
        pred_coords = pred_coordinates.cpu()
        # gt_coords = torch.flip(gt_coords, dims=[-1])
        # pred_coords = torch.flip(pred_coords, dims=[-1])
        gt_coords = gt_coords.numpy()
        pred_coords = pred_coords.numpy()
        diagonal = torch.norm(torch.Tensor([*imgs.shape[2:]]), p=2)
        marker_size = int(0.02 * diagonal) #TODO: extract percentage param to config?
        line_size = max(1,int(0.005 * diagonal)) #TODO: extract percentage param to config?
        for i in range(imgs.shape[0]):
            img = images[i, ...].cpu().numpy().transpose(1, 2, 0) * 255.0
            img = img.copy().astype(np.uint8) if img.shape[2] > 1 else cv2.cvtColor(img.copy().astype(np.uint8), cv2.COLOR_GRAY2RGB)
            for coords, color in zip(
                [gt_coords, pred_coords],
                [gt_color, pred_color]
            ):       
                coord_i = np.int32(coords[i, ...])
                for k, coord in enumerate(coord_i):
                    if marker < 0:
                        cv2.circle(img, 
                            tuple(coord), 
                            marker_size, color.tolist(), thickness=line_size
                        )
                    else:
                        cv2.drawMarker(img, 
                            tuple(coord),
                            color.tolist(),
                            marker, marker_size, line_size
                        )
                imgs[i, ...] = img.transpose(2, 0, 1)
        visdom.images(
            imgs,
            #np.flip(imgs, axis=1), #NOTE: Why flip?
            win=win,
            env=env,
            opts={
                'title': key,
                'caption': key,
                'jpgquality': 100,
            }
        )