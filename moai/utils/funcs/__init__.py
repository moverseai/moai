import typing

def passthrough(*args: typing.Any) -> typing.Any:
    return args[0] if len(args) == 1 else args