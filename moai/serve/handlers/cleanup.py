import logging
import os
import typing
from collections.abc import Callable

log = logging.getLogger(__name__)


class CleanUp(Callable):
    def __init__(self, dirs: typing.List[str]) -> None:
        self.dirs = [dirs] if type(dirs) is str else dirs  # directories to be deleted
        """
        Responsible for deleting data from Docker worker.

        Args:
            dirs (typing.List[str]): Directories to be deleted.
        """

    def __call__(
        self, data: typing.Mapping[str, typing.Any], void: typing.Any
    ) -> typing.Any:
        for folder in self.dirs:
            try:
                # check if is folder or file
                if os.path.isfile(folder):
                    os.remove(folder)
                    log.debug(f"File {folder} deleted successfully.")
                else:
                    # remove files
                    for file in os.listdir(folder):
                        os.remove(os.path.join(folder, file))
                log.debug(f"Directory {folder} deleted successfully.")
            except OSError as e:
                log.error(f"Error: {folder} : {e.strerror}")

        return [{"is_success": True, "message": "Directories deleted successfully."}]
