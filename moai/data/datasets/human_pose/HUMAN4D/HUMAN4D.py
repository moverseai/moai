from moai.data.datasets.human_pose.HUMAN4D.importers import (
	load_intrinsics_repository,
	get_intrinsics,
	load_extrinsics,
	load_depth_pgm,
	load_image,
	load_skel3D_bin,
	load_rgbd_skip,
)
from moai.data.datasets.human_pose.HUMAN4D.utils import sort_nicely
from moai.data.datasets.human_pose.HUMAN4D.enums import joint_selection
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import sys
import torch
import numpy
import logging

'''
Dataset importer. We assume that data follows the below structure.
root_path
----|device_repository.json
----|Subject1
	|pose
	----|Action 1
		----|Dump
			----|color
			----|depth
			----|gposes3d
		|
		|
	----|Action N
		----|Dump
			----|color
			----|depth
			----|gposes3d
	|
	|
----SubjectN
'''

log = logging.getLogger(__name__)

__all__ = ["HUMAN4D"]

class HUMAN4D(Dataset):
	def __init__(self, 
		path: str,
		data_types,
		center_crop,
		device_list,
		metadata_path,
		mode,
		depth_threshold,
		clip_frames
	):
		super(HUMAN4D, self).__init__()
		self.center_crop = center_crop
		self.device_list = device_list
		self.depth_threshold = depth_threshold
		self.mode = mode
		device_repo_path = os.path.join(path, "device_repository.json")
		self.data_types = data_types
		if not os.path.exists(device_repo_path):
			raise ValueError("{} does not exist, exiting.".format(device_repo_path))
		self.depth_intrinsics_repository = load_intrinsics_repository(device_repo_path, "depth")
		self.color_intrinsics_repository = load_intrinsics_repository(device_repo_path, "color")

		if not os.path.exists(path):
			raise ValueError("{} does not exist, exiting.".format(path))
		
		use_metadata = True
		if not os.path.exists(os.path.join(path, metadata_path)):
			use_metadata = False
			
		if use_metadata:
			metadata_samples = open(os.path.join(path, metadata_path), 'r').readlines()
			metadata_dict = {}
			for line in metadata_samples:
				values = line.split(' ')
				values[-1] = values[-1].strip()
				if values[1] not in metadata_dict.keys():
					metadata_dict[values[1]] = {}
				if values[2] not in metadata_dict[values[1]].keys():
					metadata_dict[values[1]][values[2]] = []
				metadata_dict[values[1]][values[2]].append(int(values[0]))

		self.data = {}
		self.extrinsics_paths_per_subject = {}
		subject_folders = [subject for subject in os.listdir(path) if "." not in subject and len(subject) == 2]
		# iterate over each recorded folder
		for subject in subject_folders:
			if use_metadata and subject not in metadata_dict.keys():
				continue

			abs_subject_path = os.path.join(path, subject)
			if not os.path.isdir(abs_subject_path):
				continue			

			subject_recordings = [sub_rec for sub_rec in os.listdir(abs_subject_path) if '-' in sub_rec and '!' not in sub_rec and '.' not in sub_rec]
			# We remove and hold separately the pose folder for the Calibration
			# subject_recordings.pop(subject_recordings.index("pose"))
			pose_folder = os.path.join(abs_subject_path, "pose")
			if not os.path.exists(pose_folder):
				log.warning("Folder {} does not contain \"pose\" folder".format(abs_subject_path))
			
			for recording in subject_recordings:
				if use_metadata and recording not in metadata_dict[subject].keys():
					continue	

				recording_modalities = os.path.join(abs_subject_path, recording, "Dump")
				skel_3d_gt_directory = os.path.join(recording_modalities, "gposes3d")				
				
				skel_3d = os.listdir(skel_3d_gt_directory)
				sort_nicely(skel_3d)
				skipped = load_rgbd_skip(os.path.join(abs_subject_path, "offsets.txt"), recording)

				if "color" in data_types:
					color_directory = os.path.join(recording_modalities, "color")
					color_img_paths = os.listdir(color_directory)
					sort_nicely(color_img_paths)
					color_img_paths = color_img_paths[4 * (skipped + 1):]


					for color_img_path in color_img_paths:
						full_filename = os.path.join(color_directory, color_img_path)
						_, ext = os.path.splitext(full_filename)
						
						if ext != ".png":
							continue
						
						if self.mode == "single":
							_id, _name,_type,_ = color_img_path.split("_")

							if use_metadata and int(_id) not in metadata_dict[subject][recording]:
								continue

							unique_name = subject + "-" + recording + "-" + _name + "-" + str(int(_id) - skipped)

							#skip names that we dp now want to load
							if _name not in device_list:
								continue
							
							if unique_name not in self.data:
								self.data[unique_name] = {}						
							
							if "pose" not in self.data[unique_name]:
								self.data[unique_name]['pose'] = os.path.join(pose_folder, _name + ".extrinsics")

							if "dev_name" not in self.data[unique_name]:
								self.data[unique_name]['dev_name'] = _name

							self.data[unique_name][_type] = full_filename
							
						else:
							_id, _name, _type, _ = color_img_path.split("_")
							if use_metadata and int(_id) not in metadata_dict[subject][recording]:
								continue
							unique_name = subject + "-" + recording + "-" + str(int(_id) - skipped)

							#skip names that we dp now want to load
							if _name not in device_list:
								continue
							
							if unique_name not in self.data:
								self.data[unique_name] = {}						

							if _name not in self.data[unique_name]:
								self.data[unique_name][_name] = {}
							
							if "pose" not in self.data[unique_name]:
								self.data[unique_name][_name]['pose'] = os.path.join(pose_folder, _name + ".extrinsics")
							
							self.data[unique_name][_name][_type] = full_filename
				
				if "depth" in data_types:
					depth_directory = os.path.join(recording_modalities, "depth")
					depth_img_paths = os.listdir(depth_directory)
					sort_nicely(depth_img_paths)
					depth_img_paths = depth_img_paths[4 * (skipped + 1):]
					for depth_img_path in depth_img_paths:
						full_filename = os.path.join(depth_directory, depth_img_path)
						_, ext = os.path.splitext(full_filename)
						
						if ext != ".pgm":
							continue

						if self.mode == "single":
							_id, _name, _type,_ = depth_img_path.split("_")
							if use_metadata and int(_id) not in metadata_dict[subject][recording]:								
								continue

							unique_name = subject + "-" + recording + "-" + _name + "-" + str(int(_id) - skipped)
							#skip names that we dp now want to load
							if _name not in device_list:
								continue
							
							if unique_name not in self.data:
								self.data[unique_name] = {}						
							
							if "pose" not in self.data[unique_name]:
								self.data[unique_name]['pose'] = os.path.join(pose_folder, _name + ".extrinsics")
							
							if "dev_name" not in self.data[unique_name]:
								self.data[unique_name]['dev_name'] = _name

							self.data[unique_name][_type] = full_filename
						else:
							_id, _name,_type,_ = depth_img_path.split("_")
							if use_metadata and int(_id) not in metadata_dict[subject][recording]:
								continue
							unique_name = subject + "-" + recording + "-" + str(int(_id) - skipped)
							#skip names that we dp now want to load
							if _name not in device_list:
								continue
							
							if unique_name not in self.data:
								self.data[unique_name] = {}						

							if _name not in self.data[unique_name]:
								self.data[unique_name][_name] = {}
							
							if "pose" not in self.data[unique_name]:
								self.data[unique_name][_name]['pose'] = os.path.join(pose_folder, _name + ".extrinsics")
		
							self.data[unique_name][_name][_type] = full_filename
				#Always load the GT data. 
				### Keep the unique name consistent in all loading scenarios				
				for skel_3d_frame in skel_3d:
					full_filename = os.path.join(skel_3d_gt_directory, skel_3d_frame)
					_id, ext = os.path.splitext(skel_3d_frame)
					if ext != ".npy":
						continue
					if self.mode == "single":
						for _name in self.device_list:
							unique_name = subject + "-" + recording + "-" + _name + "-" + _id
							# this check also satisfies that we get data only from the selected frames in "metadata_dict"
							if unique_name not in self.data:
								continue
							if "joints_3d_raw" not in self.data[unique_name]:
								self.data[unique_name]["joints_3d_raw"] = full_filename
					else:
						unique_name = subject + "-" + recording + "-" + _id

						if unique_name not in self.data:
							continue

						if "joints_3d_raw" not in self.data[unique_name]:
							self.data[unique_name]["joints_3d_raw"] = full_filename				
		log.info(metadata_path.split('.')[0] + " is succesfully loaded.")
			

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		#get an entry
		key = list(self.data.keys())[idx]
		datum = self.data[key]
		datum_out = {}
		### JOINTS 3D
		joints_3d_raw = load_skel3D_bin(datum["joints_3d_raw"], scale= 1e-03)
		joints_3d = torch.cat([joints_3d_raw[:, :, joint_id, :] for joint_id in joint_selection] , dim=0).unsqueeze(0).permute(0,2,1,3)
		datum_out.update({"joints_3d" : joints_3d.squeeze(0).squeeze(0)})
		if self.mode == "single":
			#Calibration extrinsics
			rot, trans, inv_rot, inv_trans = load_extrinsics(\
				datum["pose"])
			if "color" in self.data_types and not "depth" in self.data_types:
				color_tensor = load_image(datum["color"])
				color_intrinsics, color_intrinsics_inv = get_intrinsics(\
					datum["dev_name"], self.color_intrinsics_repository, 1, self.center_crop)
				## DATUM UPDATE
				datum_out.update({
					"color" : color_tensor.squeeze(0),
					"cintr" : color_intrinsics,
					"dintr" : depth_intrinsics,
					"depth" : masked_depth_tensor.squeeze(0), 
					"campose_rotation" : inv_rot, 
					"campose_translation" :  inv_trans
					})
			elif "depth" in self.data_types and not "color" in self.data_types:				
				depth_tensor = load_depth_pgm(datum["depth"], scale = 1e-04)
				mask = (depth_tensor < self.depth_threshold).type(torch.float32)
				masked_depth_tensor = depth_tensor * mask
				depth_intrinsics, depth_intrinsics_inv = get_intrinsics(\
					datum["dev_name"], self.depth_intrinsics_repository, 4, self.center_crop)

				depth_intrinsics_down4, depth_intrinsics_inv_down4 = get_intrinsics(\
					datum["dev_name"], self.depth_intrinsics_repository, 16, self.center_crop)
				## DATUM UPDATE
				datum_out.update({
					"dintr" : depth_intrinsics,
					"dintr_down4" : depth_intrinsics_down4,
					"depth" : masked_depth_tensor.squeeze(0), 
					"campose_rotation" : inv_rot, 
					"campose_translation" :  inv_trans
					})
			elif "depth" in self.data_types and "color" in self.data_types:
				color_tensor = load_image(datum["color"])
				color_intrinsics, color_intrinsics_inv = get_intrinsics(\
					datum["dev_name"], self.color_intrinsics_repository, 1, self.center_crop)
				depth_tensor = load_depth_pgm(datum["depth"], scale = 1e-04)
				mask = (depth_tensor < self.depth_threshold).type(torch.float32)
				masked_depth_tensor = depth_tensor * mask
				depth_intrinsics, depth_intrinsics_inv = get_intrinsics(\
					datum["dev_name"], self.depth_intrinsics_repository, 4, self.center_crop)
				## DATUM UPDATE
				datum_out.update({
					"color" : color_tensor.squeeze(0),
					"cintr" : color_intrinsics,
					"dintr" : depth_intrinsics,
					"depth" : masked_depth_tensor.squeeze(0), 
					"campose_rotation" : inv_rot, 
					"campose_translation" :  inv_trans
					})
		else:
			for device in self.device_list:
				color_tensor = load_image(datum[device]["color"])
				color_intrinsics, color_intrinsics_inv = get_intrinsics(\
				device, self.color_intrinsics_repository, 1)

				depth_tensor = load_depth_pgm(datum[device]["depth"], scale = 1e-04)
				mask = (depth_tensor < self.depth_threshold).type(torch.float32)
				masked_depth_tensor = depth_tensor * mask
				depth_intrinsics, depth_intrinsics_inv = get_intrinsics(\
				device, self.depth_intrinsics_repository, 4)
				#Calibration extrinsics
				rot, trans, inv_rot, inv_trans = load_extrinsics(\
					datum[device]["pose"])
				## DATUM UPDATE
				datum_out.update({
					device + "_color" : color_tensor.squeeze(0),
					device + "_cintr" : color_intrinsics,
					device + "_dintr" : depth_intrinsics,
					device + "_depth" : masked_depth_tensor.squeeze(0), 
					device + "_campose_rotation" : inv_rot, 
					device + "_campose_translation" :  inv_trans,
					})
		
		return datum_out

	def get_data(self):
		return self.data