import torch
import toolz
import colour
import logging
import typing
import numpy as np
import open3d as o3d
import cv2
import omegaconf.omegaconf
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)


from collections.abc import Callable

try:
    import rerun as rr
except:
    log.error(f"Please `pip install rerun-sdk` to use ReRun visualisation module.")


class RRLog(Callable):
    r"""
    This logger should support all the different types of rr logging.
    """

    __funtion_map = {
        'LBU': rr.ViewCoordinates.LBU,
        'LBD': rr.ViewCoordinates.LEFT_HAND_Z_DOWN,
        'RUF': rr.ViewCoordinates.RUF,
    }

    def __init__(
        self,
        name: str,  # name of the logger
        types: typing.Union[
            str, typing.Sequence[str]
        ],  # the different types of logging
        keys: typing.Union[str, typing.Sequence[str]],  # the different keys to log
        colors: typing.Union[str, typing.Sequence[str]],
        transforms: typing.Union[str, typing.Sequence[str]],
        paths_to_hierarchy: typing.Sequence[str],
        skeleton: typing.Sequence[int] = None,
        export_path: str = None,
        world_coordinates: str = "RUF", # x is right, y is up, z is forward
    ):
        rr.init(name)
        rr.spawn() if export_path is None else rr.save(export_path)
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.types = [types] if isinstance(types, str) else list(types)
        self.paths_to_hierarchy = (
            [paths_to_hierarchy]
            if isinstance(paths_to_hierarchy, str)
            else list(paths_to_hierarchy)
        )
        self.colors = (
            [colour.Color(colors)]
            if isinstance(colors, str)
            else list(colour.Color(c) if isinstance(c, str) else None for c in colors)
        )
        self.transforms = (
            transforms if isinstance(transforms, str) else list(transforms)
        )
        self.access = lambda td, k: toolz.get_in(k.split("."), td)
        self.step = 0
        self.kintree = skeleton
        self.log_map = {
            "color": self._image,
            "points": self._points,
            "mesh": self._mesh,
            "pose": self._pose,
            "depth": self._depth,
            "infrared": self._depth,
            "scalar": self._scalar,
            "camera": self._log_pinhole_camera,
            "transform": self._log_transform,
            "openpose": self._openpose, # only for debugging
        }
        self.transform_map = {
            None: lambda x: x,  # pass through
            "flip": self._flip,
            "rotate": self._rotate,
            "rotate_kpts": self._rotate_kpts,
            "transpose": self._transpose,
        }
        # log world camera coordinates system
        # rr.log_view_coordinates("world", up="+Y", right_handed=True, timeless=True)
        # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, timeless=True)  # X=Right, Y=Down, Z=Forward
        # test memory recoriding
        # self.rec = rr.memory_recording()
        # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)  # X=Right, Y=Down, Z=Forward
        rr.log("world", RRLog.__funtion_map[world_coordinates], timeless=True)  # X=Right, Y=Down, Z=Forward
        # floor
        self.world_coordinates = world_coordinates
        self._create_floor()

    def _create_floor(self):
        # Initialize an empty list to hold all line segments
        line_segments = []
        # find up axis and create floor plane
        def create_tile_xy(x, y):
            """Creates a single tile at position (x, y)."""
            return [
                [[x, y, 0], [x + 1, y, 0]],  # Bottom edge
                [[x + 1, y, 0], [x + 1, y + 1, 0]],  # Right edge
                [[x + 1, y + 1, 0], [x, y + 1, 0]],  # Top edge
                [[x, y + 1, 0], [x, y, 0]],  # Left edge
            ]
        def create_tile_xz(x, z):
            """Creates a single tile at position (x, z) in the X-Z plane."""
            return [
                [[x, 0, z], [x + 1, 0, z]],  # Front edge (along X-axis)
                [[x + 1, 0, z], [x + 1, 0, z + 1]],  # Right edge (along Z-axis)
                [[x + 1, 0, z + 1], [x, 0, z + 1]],  # Back edge (back along X-axis)
                [[x, 0, z + 1], [x, 0, z]],  # Left edge (back along Z-axis)
            ]
        if self.world_coordinates[1] == 'U':
            # y is up axis
            # floor is in xz plane
            # Create a 10x10 grid of tiles
            for x in range(-5,6):
                for y in range(-5,6):
                    line_segments.extend(create_tile_xz(x, y))
        
        elif self.world_coordinates[2] == 'U':
            # z is up axis
            # floor is in xy plane
            # Create a 10x10 grid of tiles
            for x in range(-5,6):
                for y in range(-5,6):
                    line_segments.extend(create_tile_xy(x, y))
        
        # Log the floor
        rr.log(
            "world/floor",
            rr.LineStrips3D(
                np.array(
                    line_segments
                ),
                colors=[128,128,128] # grey color
            ),
            timeless=True,
        )



    @staticmethod
    def _flip(x, axis, width=None, height=None):
        if height is not None:
            y = np.flip(x.copy(), axis)
            # x[:, :, 0] = width - x[:, :, 0]
            y[:, 1, :] = height - y[:, 1, :]
        else:
            y = np.flip(x.copy(), axis)
        return y

    @staticmethod
    def _rotate_kpts(kpts, rotate_angle):
        # y = np.flip(kpts.copy(), 1)
        # y[:, 1, :] = 1280 - y[:, 1, :]
        rotate_angle = np.deg2rad(rotate_angle)
        # Create a rotation matrix
        rotation_matrix = np.array(
            [
                [np.cos(rotate_angle), -np.sin(rotate_angle)],
                [np.sin(rotate_angle), np.cos(rotate_angle)],
            ]
        )

        rotated_keypoints = np.dot(rotation_matrix.T, kpts).transpose(1, 0, 2)

        return rotated_keypoints

    @staticmethod
    def _rotate(x, cv2_rotate_code):
        # ROTATE_90_CLOCKWISE        = 0,
        # ROTATE_180                 = 1,
        # ROTATE_90_COUNTERCLOCKWISE = 2,
        b = x.shape[0]
        images = []
        for i in range(b):
            cv2_image = x[i].transpose(1, 2, 0) if len(x.shape) == 4 else x[i]
            cv2_image = cv2.rotate(cv2_image, cv2_rotate_code)
            images.append(cv2_image.transpose(2, 0, 1))  # return to torch image format

        return np.stack(images, axis=0)
    
    @staticmethod
    def _transpose(x, axis=1):
        b = x.shape[0]
        images = []
        for i in range(b):
            if axis == 0:
                cv2_image = x[i].transpose(1, 2, 0) if len(x.shape) == 4 else x[i]
            else:
                cv2_image = x[i].transpose(0, 2, 1) if len(x.shape) == 4 else x[i]
            images.append(cv2_image)  # return to torch image format

        return np.stack(images, axis=0)


    @staticmethod
    def _image(
        image: np.ndarray,
        path: str,
        c: colour.Color,  # not used
        jpeg_quality: int = 5,
    ) -> None:
        rr.log_image(
            path,
            np.ascontiguousarray(image[0].transpose(1, 2, 0) * 255, np.uint8),
            jpeg_quality=jpeg_quality,
        )
        # rr.log_image(path, image.transpose(0, 2, 3, 1), jpeg_quality = jpeg_quality)

    @staticmethod
    def _depth(
        depth: np.ndarray,
        path: str,
        c: colour.Color,  # not used
    ) -> None:
        # TODO: handle batch
        rr.log_depth_image(path, depth)

    @staticmethod
    def _scalar(points: np.ndarray, path: str, c: colour.Color) -> None:
        if len(points.shape) == 3:
            rr.log_scalar(
                path, points[0][0][0], color=c.get_rgb()
            ) if points.ndim != 0 else rr.log_scalar(path, points, color=c.get_rgb())
        else:
            rr.log_scalar(
                path, points[0], color=c.get_rgb()
            ) if points.ndim != 0 else rr.log_scalar(path, points, color=c.get_rgb())

    def _openpose(
        self,
        kpts: np.ndarray,
        path: str,
        c: colour.Color,
    ) -> None:
        openpose_tree = [
            # Right leg
            [8, 9], [9, 10], [10, 11], 
            [11, 22], [22, 23], [11, 24], 
            # Left leg
            [8, 12], [12, 13], [13, 14],
            [14, 19], [19, 20], [14, 21],
            # Upper body
            [8, 1], 
            # Head
            [1, 0],
            # Right arm
            [1, 2], [2, 3], [3, 4],
            # Left arm
            [1, 5], [5, 6], [6, 7],
            # Face
            [0, 15], [15, 17], [0, 16], [16, 18],
        ]
        labels = [
            # Right leg
            'Central Hip to Right Hip', 'Right Hip to Right Knee', 'Right Knee to Right Ankle',
            'Right Ankle to Right Big Toe', 'Right Big Toe to Right Small Toe', 'Right Ankle to Right Heel',
            # Left leg
            'Central Hip to Left Hip', 'Left Hip to Left Knee', 'Left Knee to Left Ankle',
            'Left Ankle to Left Big Toe', 'Left Big Toe to Left Small Toe', 'Left Ankle to Left Heel',
            # Upper body
            'Central Hip to Neck',
            # Head
            'Neck to Nose',
            # Right arm
            'Neck to Right Shoulder', 'Right Shoulder to Right Elbow', 'Right Elbow to Right Wrist',
            # Left arm
            'Neck to Left Shoulder', 'Left Shoulder to Left Elbow', 'Left Elbow to Left Wrist',
            # Face
            'Nose to Right Eye', 'Right Eye to Right Ear', 'Nose to Left Eye', 'Left Eye to Left Ear',
        ]
        colors = [
            # Right leg
            [255, 0, 0], [255, 0, 0], [255, 0, 0],
            [255, 0, 0], [255, 0, 0], [255, 0, 0],
            # Left leg
            [0, 0, 255], [0, 0, 255], [0, 0, 255],
            [0, 0, 255], [0, 0, 255], [0, 0, 255],
            # Upper body
            [0, 255, 0],
            # Head
            [0, 255, 0],
            # Right arm
            [255, 255, 0], [255, 255, 0], [255, 255, 0],
            # Left arm
            [0, 255, 255], [0, 255, 255], [0, 255, 255],
            # Face
            [255, 0, 255], [255, 0, 255], [255, 0, 255], [255, 0, 255],
        ]
                        
        b = kpts.shape[0]
        if (
            kpts.shape[-1] == 3 or kpts.shape[-1] == 2
        ):
            pass
        else:
            kpts = kpts.transpose(0, 2, 1)
        for b_ in range(b):
            edges = []
            for nk, (i, j) in enumerate(openpose_tree):
                # check if keypoints are within the image
                edges.append([kpts[b_][i], kpts[b_][j]])
                # edges.append(kpts[b_][j])
            rr.log(
                path,
                rr.LineStrips2D(
                    edges,
                    colors=colors,
                    labels=labels,
                )
            )
            # rr.log_line_segments(path, edges, color=c.get_rgb())
            # rr.log_points(path, kpts[b_], colors=c.get_rgb())
    
    def _pose(
        self,
        kpts: np.ndarray,
        path: str,
        c: colour.Color,
        kinematic_tree: typing.Sequence[int] = None,
    ) -> None:
        b = kpts.shape[0]
        if (
            kpts.shape[-1] == 3 or kpts.shape[-1] == 2
        ):  # expect 3D keypoints of [N, 3] or [N, 2]
            pass
            # kpts = kpts.reshape(b, kpts.shape[-1], -1)
        else:
            # kpts = kpts.reshape(b, kpts.shape[-1], -1)
            kpts = kpts.transpose(0, 2, 1)
        for b_ in range(b):
            edges = []
            for nk, (i, j) in enumerate(self.kintree):
                edges.append(kpts[b_][i])
                edges.append(kpts[b_][j])
            rr.log_line_segments(path, edges, color=c.get_rgb())
            rr.log_points(path, kpts[b_], colors=c.get_rgb())

    @staticmethod
    def _points(
        points: np.ndarray,
        path: str,
        c: colour.Color,
    ) -> None:
        b = points.shape[0]
        # if (
        #     points.shape[-1] == 3 or points.shape[-1] == 2
        # ):  # expect 3D keypoints of [N, 3] or [N, 2]
        #     pass
        # else:
        #     points = points.transpose(0, 2, 1)
        # expects [B, N, 3]
        if points.shape[1] == 2 or points.shape[1] == 3:
            # means input is [B,2,N] or [B,3,N]
            points = points.transpose(0, 2, 1)
        else:
            # means input is [B,N,2] or [B,N,3]
            pass
        for i in range(b):
            rr.log_points(path, points[i], colors=c.get_rgb())

    # @staticmethod
    def _log_transform(
        self,
        transform: np.ndarray,
        path: str,
        xyz: str = "RDF",
    ) -> None:
        # quat = R.from_matrix(transform[0][:3, :3]).as_quat()
        # breakpoint()
        # rr.log(path, rr.TranslationAndMat3(
        #     # child_from_parent = (transform[0][:3,3], quat),
        #     # xyz = 'RDF',
        #     translation = transform[0][:3,3],
        #     matrix = R.from_matrix(transform[0][:3,:3]).as_matrix(),
        #     from_parent = True,
        #     )
        # )
        rr.set_time_sequence("frame_nr", self.step)
        rr.log(
            path,
            rr.Transform3D(
                translation=transform[0][:3, 3],
                # mat3x3=R.from_matrix(transform[0][:3, :3]).as_matrix(),
                mat3x3=transform[0][:3, :3],
                from_parent = True,
            ),
        )
        rr.log(path, rr.ViewCoordinates.RDF, timeless=True)  # X=Right, Y=Down, Z=Forward
        #rr.log(path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)  # X=Right, Y=Forward, Z=Up

    @staticmethod
    def _log_pinhole_camera(
        camera_matrix: np.ndarray,
        path: str,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        # rr.log_pinhole(
        #     path,
        #     child_from_parent=camera_matrix[0],
        #     width=1920,  # 2048, #1280, #1920, #1280,
        #     height=1080,  # 2448, #720, #1080, #720,
        # )
        rr.log(
            path,
            rr.Pinhole(
                # width = 1920,
                # height = 1080,
                # width= 1280,
                # height= 720,
                image_from_camera = camera_matrix,
            )
        )

    def _mesh(
        self,
        array: typing.List[np.ndarray],
        # faces: np.ndarray,
        path: str,
        c: colour.Color,
    ) -> None:
        b = array[0].shape[0]
        for i in range(b):
            vertices, faces = array
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices[i]),
                o3d.utility.Vector3iVector(faces[i]),
            )
            mesh.compute_vertex_normals()  # Needs to calculate normals to apply apply color
            rr.log_mesh(
                path,
                vertices,
                indices=faces,
                normals=np.asarray(mesh.vertex_normals),
                albedo_factor=c.get_rgb(),
            )
            del mesh

    def __call__(
        self, tensors: typing.Dict[str, torch.Tensor], step: typing.Optional[int] = None
    ) -> None:
        # print("RRLog called")
        # iterate over the different types of logging
        # set time sequence
        rr.set_time_sequence("frame_nr", self.step)
        for t, k, c, path, tr in zip(
            self.types, self.keys, self.colors, self.paths_to_hierarchy, self.transforms
        ):  
            # call the correct logging function from the map
            self.log_map[t](
                self.transform_map[toolz.get_in(["type"], tr)](
                    [self.access(tensors, k_).detach().cpu().numpy() for k_ in k]
                    if isinstance(k, omegaconf.ListConfig)
                    else self.access(tensors, k).detach().cpu().numpy(),
                    **tr.args if tr is not None else {},
                ),
                path,
                c,
            )
        # increment the step
        self.step += 1
