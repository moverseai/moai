from moai.data.datasets.object_pose.UAVA.importers import (
    load_image,
    load_depth,
    load_normal,
)

from PIL import Image

import torch
import os
import sys
import torch
import json
import logging
import typing

log = logging.getLogger(__name__)

__all__ = ["UAVA"]

class UAVA(torch.utils.data.Dataset):
    def __init__(self,
        root:               str, # path to root folder
        metadata:           str, # path to metadata .json
        split:              str, # [train,test,val]
        drones:             typing.List[str],
        views:              typing.List[str],
        frames:             typing.List[str],
        types:              typing.List[str],
    ):
        super(UAVA,self).__init__()
        self.drones = drones
        self.views = views
        self.frames = frames
        self.types = types
        self.data = {}
        buildings = self.get_splits(split, root)
        if not os.path.exists(metadata):
            error_message = f"Metadata annotations file ({metadata}) not found."
            log.error(error_message)
            raise ValueError(error_message)
        with open(metadata) as f:
            self.annotations = json.load(f)
        for drone in os.listdir(root):
            if drone not in drones:
                log.warning(f"Skipping {drone} drone model as it is not included in the selected drones list.")
                continue
            drone_path = os.path.join(root, drone)
            for view in os.listdir(drone_path):
                if view not in views:
                    log.warning(f"Skipping {view} view as it is not included in the selected views list.")
                    continue
                view_path = os.path.join(drone_path, view)
                for data_type in os.listdir(view_path):
                    if data_type not in types:
                        log.warning(f"Skipping {data_type} data type as it is not included in the selected data types list.")
                        continue
                    for building in buildings:
                        img_path = os.path.join(view_path, data_type, building)
                        for img in os.listdir(img_path):
                            if (len(img.split("_")) <= 6):
                                row, date , frame , __ , view , _ = img.split("_")
                            else:
                                row, date , frame , __ , _, view , ___ = img.split("_")                            
                            if view not in views:
                                continue                            
                            if int(frame) not in frames:
                                continue
                            full_img_name = os.path.join(img_path, img)                            
                            unique_name = building + "_" + str(row) + "_" + str(date)
                            try:
                                if self.annotations['drones'][drone][unique_name]['exocentric']["0"]['visibleAll'] == False:
                                    continue
                            except KeyError:
                                continue
                            if unique_name not in self.data:
                                self.data[unique_name] = {}
                            if view not in self.data[unique_name]:
                                self.data[unique_name][view] = {}
                            if frame not in self.data[unique_name][view]:
                                self.data[unique_name][view][frame] = {}
                            self.data[unique_name][view][frame][data_type] = full_img_name
 
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> typing.Dict[str, torch.Tensor]:
        key = list(self.data.keys())[idx]
        datum = self.data[key]
        datum_out = {}
        for view in self.views:
            for frame in self.frames:
                for data_type in self.types:
                    if data_type == "colour":
                        value = load_image(datum[view][str(frame)][data_type])
                    elif data_type == "silhouette" and view == "exocentric":
                        value = load_image(datum[view][str(frame)][data_type])
                    elif data_type == "colour" or data_type == "silhouette":
                          continue
                    elif data_type == "depth":
                        value = load_depth(
                            datum[view][str(frame)][data_type]
                        )
                    elif data_type == "normal":
                        value = load_normal(
                            datum[view][str(frame)][data_type]
                        )
                    else:
                        value = load_image(
                            datum[view][str(frame)][data_type]
                        )
                    datum_out.update({
                        str(view) + "_"+ str(frame) + "_" + str(data_type): value.squeeze(0)
                    })
                    if view == "exocentric":
                        pose = torch.FloatTensor(json.JSONDecoder().decode(
                            self.annotations['drones'][self.drones[0]][key][view][str(frame)]['pose']
                        )).reshape(4,4)
                        pose_inv = pose.inverse()
                        bbox_2d = torch.FloatTensor(json.JSONDecoder().decode(
                            self.annotations['drones'][self.drones[0]][key][view][str(frame)]['2dbbox']
                        ))
                        bbox_3d = torch.FloatTensor(json.JSONDecoder().decode(
                            self.annotations['drones'][self.drones[0]][key][view][str(frame)]['3dbbox']
                        )).reshape(8,2)
                        points_3d = torch.FloatTensor(json.JSONDecoder().decode(
                            self.annotations['drones'][self.drones[0]][key][view][str(frame)]['3dpoints']
                        )).reshape(9,2)
                        key_root = str(view) + "_"+ str(frame) + "_" 
                        datum_out.update({key_root + "pose" : pose})
                        datum_out.update({key_root + "pose_inv" : pose_inv})
                        datum_out.update({key_root + "2dbbox" : bbox_2d})
                        datum_out.update({key_root + "3dbbox" : bbox_3d})
                        datum_out.update({key_root + "points_3d" : points_3d})
                        datum_out.update({key_root + "key" : key})
                        datum_out.update({key_root + "translation" : pose[:3,3]})
                        datum_out.update({key_root + "rotation" : pose[:3,:3]})
                    else:
                        src = 0 if int(frame) == 1 else 1
                        ego_pose = torch.FloatTensor(json.JSONDecoder().decode(
                            self.annotations['drones'][self.drones[0]][key][view][str(src)]['pose']
                        )).reshape(4,4)
                        key_root = str(view) + "_"+ str(frame) + "_" 
                        datum_out.update({key_root + "key" : key})
                        datum_out.update({key_root + 'source_to_target' : ego_pose})
        return datum_out

    def get_splits(self, split: str, root: str) -> typing.List[str]:
        if split == "train":
            txt_path = os.path.join(root, "data splits", "scenes_train.txt")
        elif split == "test":
            txt_path = os.path.join(root, "data splits", "scenes_test.txt")
        elif split == "val":
            txt_path = os.path.join(root, "data splits", "scenes_val.txt")
        with open(txt_path,'r') as txt_file:
            splits = [line.strip() for line in txt_file]
        return splits
