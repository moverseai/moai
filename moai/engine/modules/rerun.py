import logging
import typing
from uuid import uuid4

import colour
import numpy as np
import toolz
from omegaconf.omegaconf import DictConfig

from moai.utils.color.colormap import random_color
from moai.utils.funcs import get

try:
    import rerun as rr
except:
    from pytorch_lightning.core.module import warning_cache

    warning_cache.warn(f"Please `pip install rerun-sdk` to use rerun visualisation.")

__all__ = ["Rerun"]


class RerunFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__("rerun")

    def filter(self, record):
        record.msg = record.msg.removeprefix(":moai: ")
        return record


class Rerun:
    r"""
    This logger should support all the different types of rr logging.
    """

    __COORD_SYSTEM_MAP__ = {
        "LBU": rr.ViewCoordinates.LBU,
        "LBD": rr.ViewCoordinates.LEFT_HAND_Z_DOWN,
        "RUF": rr.ViewCoordinates.RUF,
        "RFU": rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        "RDD": rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
        "LDF": rr.ViewCoordinates.LEFT_HAND_Y_DOWN,
        "RDF": rr.ViewCoordinates.RDF,
        "RDB": rr.ViewCoordinates.RDB,
        "LDB": rr.ViewCoordinates.LDB,
        "RBD": rr.ViewCoordinates.RBD,
    }

    __LABEL_TO_CLASS_ID__ = {}

    def __init__(
        self,
        name: str,
        export: typing.Optional[typing.Union[bool, str]] = None,
        recording_id: typing.Optional[str] = None,
        annotations: typing.Mapping[str, typing.Any] = None,
        world_coordinates: str = "RUF",
        add_floor: bool = True,
        root: str = "/",
        memory_limit: str = "75%",
        log: DictConfig = {},
    ) -> None:
        # NOTE: https://github.com/iterative/dvc/issues/9731
        rr.init(name, recording_id=recording_id or uuid4())
        if export is not None:
            if isinstance(export, bool) and export:
                export_path = "."
            elif isinstance(export, str):
                export_path = export
            rr.save(export_path)
        else:
            rr.spawn(memory_limit=memory_limit)
        self.world_coordinates = world_coordinates
        rr.log(root, Rerun.__COORD_SYSTEM_MAP__[world_coordinates], timeless=True)
        parents = annotations.parents if annotations is not None else None
        labels = annotations.labels if annotations is not None else None
        if parents is not None:
            self._create_annotation_context(root, parents, labels)
        plots = annotations.plots
        if plots is not None:
            self._create_scalar_plots(root, plots)
        if add_floor:
            self._create_floor(root)
        # NOTE: add more global gizmos (e.g. tr-axis)
        if handler_name := get(log, "handler"):
            handler = rr.LoggingHandler(handler_name)
            handler.addFilter(RerunFilter())
            logging.getLogger().addHandler(handler)
            if level := get(log, "level"):
                if isinstance(level, str):
                    level = level.upper()
                level = logging.getLevelName(level)
                logging.getLogger().setLevel(level)

    def _create_scalar_plots(self, root: str, plots) -> None:
        for plot in plots:
            keypath = (
                f"/plots/{plot.key}" if root == "/" else f"{root}/plots/{plots.key}"
            )
            color = plot.get("color")
            if color is None:
                color = random_color(True)
            else:
                color = colour.Color(color).get_rgb()
            rr.log(keypath, rr.SeriesLine(color=color, name=plot.key), timeless=True)

    def _create_floor(self, root: str):
        line_segments = []  # Initialize an empty list to hold all line segments

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

        if self.world_coordinates[1] == "U" or self.world_coordinates[1] == "D":
            # y is up axis, floor is in xz plane
            for x in range(-5, 6):  # Create a 10x10 grid of tiles
                for y in range(-5, 6):
                    line_segments.extend(create_tile_xz(x, y))
        elif self.world_coordinates[2] == "U":
            # z is up axis, floor is in xy plane
            for x in range(-5, 6):  # Create a 10x10 grid of tiles
                for y in range(-5, 6):
                    line_segments.extend(create_tile_xy(x, y))
        # Log the floor
        path = "/floor" if root == "/" else f"{root}/floor"
        rr.log(
            path,
            rr.LineStrips3D(
                np.array(line_segments), colors=[128, 128, 128]  # grey color
            ),
            timeless=True,
        )

    def _create_annotation_context(self, root, parents, labels):
        classes = []
        for i, (k, ps) in enumerate(parents.items()):
            Rerun.__LABEL_TO_CLASS_ID__[k] = i
            label_ids = labels[k]
            color = random_color(True) / 255.0
            classes.append(
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=i, label=k, color=color),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=j)
                        for id, j in enumerate(label_ids)
                    ],
                    keypoint_connections=toolz.filter(
                        lambda t: t[1] > 0, enumerate(ps)
                    ),
                )
            )
        rr.log(root, rr.AnnotationContext(classes), timeless=True)
