import logging
import typing

def ensure_numeric_list(
    arg: typing.Union[int, float, typing.Sequence[int], typing.Sequence[float]],
) -> typing.List[typing.Union[int, float]]:
    return [arg] if type(arg) is int or type(arg) is float else arg

def ensure_string_list(
    arg: typing.Union[str, typing.Sequence[str]],
) -> typing.List[str]:
    return [arg] if type(arg) is str else arg

def assert_sequence_size(
    logger:         logging.Logger,
    name:           str,
    sequence:       typing.Sequence[typing.Any],
    max_size:       int,
    min_size:       int=1,    
) -> None:
    length = len(sequence)
    if length < min_size:
        logger.error(f"List {name} (length: {length}) should have at least {min_size} elements.")
    if length > max_size:
        logger.error(f"List {name} (length: {length}) should have at most {max_size} elements.")