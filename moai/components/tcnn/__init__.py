# TODO: move to engine.module
try:
    import tinycudann
except:
    from pytorch_lightning.core.module import warning_cache

    warning_cache.warn(
        f"Please install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/) to use the tcnn modules."
    )
