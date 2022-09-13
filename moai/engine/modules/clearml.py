import logging
import typing

log = logging.getLogger(__name__)

__HAS_CLEAR_ML__ = False
__CLEARML_TASK__ = None

try:
    import clearml
    __HAS_CLEAR_ML__ = True
except:
    log.error(f"Please `pip install clearml` to use the ClearML engine module.")

__all__ = ['ClearML']

def _init_task(
    project_name:       str,
    task_name:          str,
    uri:                typing.Optional[str]=None,
    tags:               typing.Optional[typing.Union[str, typing.Sequence[str]]]=None,
    connect_args:       typing.Optional[typing.Union[bool, typing.Dict[str, bool]]]=True,
    connect_libs:       typing.Optional[typing.Union[bool, typing.Dict[str, bool]]]=False,
    connect_res:        typing.Optional[typing.Union[bool, typing.Dict[str, bool]]]=True,
    # connect_logs:       typing.Optional[typing.Union[bool, typing.Dict[str, bool]]]=None,
) -> None:
    global __CLEARML_TASK__
    if __CLEARML_TASK__ is None:
        __CLEARML_TASK__ = clearml.Task.init(
            project_name=project_name,
            task_name=task_name,
            task_type=clearml.TaskTypes.training,
            reuse_last_task_id=True,
            continue_last_task=False,
            output_uri=uri,
            auto_connect_arg_parser=connect_args,
            # auto_connect_frameworks=connect_libs,
            auto_connect_frameworks={
                'matplotlib': False, 'tensorflow': False,
                'tensorboard': True, 'pytorch': connect_libs,
                'xgboost': False, 'scikit': False, 'fastai': False,
                'lightgbm': False, 'hydra': True, 'detect_repository': True,
                'tfdefines': False, 'joblib': False, 'megengine': False,
                'catboost': False,
            },
            auto_resource_monitoring=connect_res,
            # auto_connect_streams=connect_logs,
            auto_connect_streams={
                'stdout': False, 'stderr': False, 'logging': True,
            },
            deferred_init=False,
        )
        if tags:
            __CLEARML_TASK__.add_tags(tags)
        __CLEARML_TASK__.connect_configuration("config_resolved.yaml", name='hydra')
    return __CLEARML_TASK__

def _get_logger(
    project_name:       str,
    task_name:          str,
    uri:                typing.Optional[str]=None,
    tags:               typing.Optional[typing.Union[str, typing.Sequence[str]]]=None,
) -> clearml.Logger:
    global __CLEARML_TASK__
    if __CLEARML_TASK__ is None:
        _init_task(project_name, task_name, uri, tags)
    return __CLEARML_TASK__.get_logger()

class ClearML(object):
    def __init__(self,
        project_name:       str,
        task_name:          str,
        uri:                typing.Optional[str]=None,
        tags:               typing.Optional[typing.Union[str, typing.Sequence[str]]]=None,
        connect_args:       typing.Optional[typing.Union[bool, typing.Dict[str, bool]]]=None,
        connect_libs:       typing.Optional[typing.Union[bool, typing.Dict[str, bool]]]=None,
        connect_res:        typing.Optional[typing.Union[bool, typing.Dict[str, bool]]]=None,
        # connect_logs:       typing.Optional[typing.Union[bool, typing.Dict[str, bool]]]=None,
    ) -> None:       
        _init_task(
            project_name, task_name, uri, tags, 
            connect_args, connect_libs, connect_res
        )