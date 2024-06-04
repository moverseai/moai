import logging
import os


def assert_path(
    logger: logging.Logger,
    name: str,
    path: str,
) -> None:
    if not os.path.exists(path):
        logger.error(f"Path ({path}) does not exist.")


def ensure_path(
    logger: logging.Logger,
    name: str,
    path: str,
) -> None:
    if not path or not os.path.exists(path):
        # logger.error(f"Path ({path}) does not exist, reverting to cwd ({os.getcwd()})")
        # return os.getcwd()
        logger.warning(f"Path ({path}) does not exist, creating it.")
        os.makedirs(path, exist_ok=True)
        return path
    else:
        return path
