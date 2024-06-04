import logging

log = logging.getLogger(__name__)

CONTEXT = None

try:
    from moai.monads.render.pytorch3d.silhouette import Silhouette as MeshSilhouette

    __all__ = [
        "MeshSilhouette",
    ]
except:
    log.error(
        f"The PyTorch3D package (https://pytorch3d.org/) is required to use the corresponding rendering monads. Install it by following the instructions in the respective page."
    )
