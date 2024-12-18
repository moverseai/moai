import logging

log = logging.getLogger(__name__)

CONTEXT = None
# TODO: move to engine.module and add log level
try:
    import nvdiffrast.torch as dr

    if CONTEXT is None:
        CONTEXT = dr.RasterizeGLContext()

    from moai.monads.render.nvdiffrast.silhouette import Silhouette as MeshSilhouette

    __all__ = [
        "MeshSilhouette",
    ]
except:
    log.error(
        f"The nvdiffrast package (https://github.com/NVlabs/nvdiffrast) is required to use the corresponding rendering monads. Install it by following the instructions in the respective repository."
    )
