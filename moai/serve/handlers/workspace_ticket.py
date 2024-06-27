import logging
import os
import typing
from collections.abc import Callable

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

import tempfile


class WorkSpaceTicketHandler(Callable):
    def __init__(
        self,
        working_dir: str,  # aboslute path to working dir
        json_key: str = "workspace_run",
    ):
        """
        Responsible for creating a working dir and adding its corresponfing path to json data
        """
        self.working_dir = working_dir
        self.json_key = json_key

        if not os.path.isabs(working_dir):
            log.error(f"working_dir ({working_dir}) is not absolute")
        os.makedirs(self.working_dir, exist_ok=True)
        log.info("Workspace ticket handler initialized")

    def __call__(
        self, json: typing.Mapping[str, typing.Any], void: typing.Any
    ) -> typing.Any:
        generated_dir_path = tempfile.mkdtemp(dir=self.working_dir)
        json[self.json_key] = generated_dir_path
        return {}  # generated_dir_path


if __name__ == "__main__":
    import os
    import platform

    if platform.system() == "Windows":
        working_dir = f"C:/Users/{os.environ['USERNAME']}/AppData/Local/Temp"
        assert os.path.isabs(working_dir)
        print(working_dir)

        data = {}
        ticket_call = WorkSpaceTicketHandler(working_dir)
        tmp_path = ticket_call(data, None)
        print(tmp_path)
        print(data)
