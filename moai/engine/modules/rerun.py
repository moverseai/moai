import typing
import toolz
import numpy as np

try:
    import rerun as rr
except:
    from pytorch_lightning.core.module import warning_cache
    warning_cache.warn(f"Please `pip install rerun-sdk` to use rerun visualisation.")

__all__ = ['Rerun']

class Rerun():
    r"""
    This logger should support all the different types of rr logging.
    """
    __funtion_map = {
        'LBU': rr.ViewCoordinates.LBU,
        'LBD': rr.ViewCoordinates.LEFT_HAND_Z_DOWN,
        'RUF': rr.ViewCoordinates.RUF,
        'RFU': rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    }

    def __init__(self,
        name: str,
        export: typing.Optional[typing.Union[bool, str]]=None,
        parents: typing.Mapping[str, typing.Sequence[int]]=None,
        labels:  typing.Mapping[str, typing.Sequence[str]]=None,
        world_coordinates: str = "RUF",
    ) -> None:
        #NOTE: https://github.com/iterative/dvc/issues/9731
        rr.init(name)
        if export is not None:
            if isinstance(export, bool) and export:
                export_path = '.'
            elif isinstance(export, str):
                export_path = export
            rr.save(export_path)
        else:
            rr.spawn()
        self.world_coordinates = world_coordinates
        rr.log("world", Rerun.__funtion_map[world_coordinates], timeless=True)
        self.parents = parents
        self.labels = labels
        if parents is not None:
            self.type_to_class_id = {}
            self._create_annotation()
        self._create_floor()
        #NOTE: add global gizmos (e.g. tr-axis)
        
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
    
    def _create_annotation(self):
        for i, (k, ps) in enumerate(self.parents.items()):
            self.type_to_class_id[k] = i
            labels = self.labels[k]
            rr.log("/", rr.AnnotationContext(rr.ClassDescription(
                info=rr.AnnotationInfo(id=i, label=k),
                keypoint_annotations=[rr.AnnotationInfo(id=id, label=j) for id, j in enumerate(labels)],
                keypoint_connections=toolz.filter(lambda t: t[1] > 0, enumerate(ps)),
            )), timeless=True)
            rr.log(k, rr.ViewCoordinates.RIGHT_HAND_Y_UP, timeless=True)
        


