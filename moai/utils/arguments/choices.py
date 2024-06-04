import logging
import typing


def assert_choices(
    logger: logging.Logger,
    name: str,
    choice: typing.Any,
    choices: typing.Sequence[typing.Any],
) -> None:
    if choice not in choices:
        logger.error(
            f"The value given for {name} ({choice}) is invalid,"
            f" acceptable values are: {choices}"
        )


def ensure_choices(
    logger: logging.Logger,
    name: str,
    choice: typing.Any,
    choices: typing.Sequence[typing.Any],
) -> None:
    if choice not in choices:
        logger.error(
            f"The value given for {name} ({choice}) is invalid,"
            f" acceptable values are: {choices}"
        )
    return choice
