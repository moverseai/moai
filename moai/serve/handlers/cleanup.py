import logging
import os
import typing
from collections.abc import Callable

log = logging.getLogger(__name__)
import shutil


class CleanUp(Callable):
    def __init__(
        self, dirs: typing.List[str], workdir_from_json: str, json_key: str
    ) -> None:
        self.dirs = [dirs] if type(dirs) is str else dirs  # directories to be deleted
        self.workdir_from_json = workdir_from_json
        self.json_key = json_key
        """
        Responsible for deleting data from Docker worker.

        Args:
            dirs (typing.List[str]): Directories to be deleted.
        """

    def __call__(
        self, data: typing.Mapping[str, typing.Any], void: typing.Any
    ) -> typing.Any:
        if self.workdir_from_json:
            input_json = void[0].get("body") or void[0].get("raw")
            folder = input_json[self.json_key]
            assert os.path.isdir(folder)
            try:
                shutil.rmtree(folder)
            except OSError as e:
                log.error(f"Error: {folder} : {e.strerror}")
            return [
                {"is_success": True, "message": "Directories deleted successfully."}
            ]

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
