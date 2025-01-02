import logging

log = logging.getLogger(__name__)

try:
    import static_ffmpeg  # isort:skip

    _HAS_STATIC_FFMPEG_ = True
except ImportError:
    log.error(
        "To use the ffmpeg engine module you need to have the static_ffmpeg package, please `pip install static-ffmpeg`"
    )
    _HAS_STATIC_FFMPEG_ = False


class FFMPEG(object):
    def __init__(
        self,
    ):
        if _HAS_STATIC_FFMPEG_:
            log.info(f"Setting up ffmpeg paths.")
            static_ffmpeg.add_paths()
