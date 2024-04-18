import typing

try:
    import rerun as rr
except:
    from pytorch_lightning.core.module import warning_cache
    warning_cache.warn(f"Please `pip install rerun-sdk` to use rerun visualisation.")

__all__ = ['Rerun']

class Rerun():
    def __init__(self,
        name: str,
        export: typing.Optional[typing.Union[bool, str]]=None,
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
        #NOTE: add floor and other global gizmos (e.g. tr-axis)
        #NOTE: add coord system
        

        


