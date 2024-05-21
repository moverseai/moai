import typing

__all__ = [
    'passthrough',
    'select',
    'select_dict',
    'select_list',
    'get',
    'get_dict',
    'get_list',
]

def passthrough(*args: typing.Any) -> typing.Any:
    return args[0] if len(args) == 1 else args

import omegaconf.omegaconf

def select(config: omegaconf.DictConfig, keypath: str) -> omegaconf.AnyNode:
    return omegaconf.OmegaConf.select(config, keypath, default=None)

def select_dict(config: omegaconf.DictConfig, keypath: str) -> omegaconf.AnyNode:
    return omegaconf.OmegaConf.select(config, keypath, default={})

def select_list(config: omegaconf.DictConfig, keypath: str) -> omegaconf.AnyNode:
    return omegaconf.OmegaConf.select(config, keypath, default=[])

import toolz

def get(config: typing.Mapping[str, typing.Any], keypath: str) -> typing.Any:
    return toolz.get_in(keypath.split('.'), config)

def get_dict(config: typing.Mapping[str, typing.Any], keypath: str) -> typing.Any:
    return toolz.get_in(keypath.split('.'), config, default={})

def get_list(config: typing.Mapping[str, typing.Any], keypath: str) -> typing.Any:
    return toolz.get_in(keypath.split('.'), config, default=[])