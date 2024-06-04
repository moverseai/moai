import itertools
import typing

import toolz

__all__ = ["repeat"]


def repeat(
    n: int, value: typing.Union[typing.Any, typing.Sequence[typing.Any]]
) -> typing.Sequence[typing.Any]:
    return list(
        itertools.repeat(value, n) if isinstance(value, list) else toolz.take(n, value)
    )
