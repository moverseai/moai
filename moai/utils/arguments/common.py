import typing
import logging

def assert_numeric(
    logger:             logging.Logger,
    name:               str,
    value:              typing.Union[float, int],
    min_value:          typing.Union[float, int]=None,  
    max_value:          typing.Union[float, int]=None,
) -> None:
    if min_value is not None and max_value is not None\
        and (value < min_value or value > max_value):
            logger.error(f"Parameter {name} (value: {value}) should be in range [{min_value}, {max_value}].")
    if min_value is not None and max_value is None and value < min_value:
            logger.error(f"Parameter {name} (value: {value}) should be larger than {min_value}.")
    if max_value is not None and min_value is None and value > max_value:
            logger.error(f"Parameter {name} (value: {value}) should be smaller than {max_value}.")

def assert_non_negative(
    logger:             logging.Logger,
    name:               str,
    value:              typing.Union[float, int],
):
    if value < 0.0:
        logger.error(f"Parameter {name} (value: {value}) should not be negative.")

def assert_negative(
    logger:             logging.Logger,
    name:               str,
    value:              typing.Union[float, int],
):
    if value >= 0.0:
        logger.error(f"Parameter {name} (value: {value}) should be negative.")
