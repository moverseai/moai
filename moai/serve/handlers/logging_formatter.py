import logging
import typing
from collections.abc import Callable

import torch

log = logging.getLogger(__name__)


class GuidLogFilter(logging.Filter):
    def __init__(self, guid: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guid = guid

    def filter(self, record: logging.LogRecord) -> bool:
        record.guid = self.guid
        return True


class SetupLogging(Callable):
    def __init__(
        self,
        input_key: str = "guid",
    ) -> None:
        super().__init__()
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s [%(guid)s]: - %(message)s"
        )
        self.input_key = input_key
        log.debug(f"GuidLogFilter: {self.input_key}")

    def __call__(
        self,
        data: typing.Mapping[str, typing.Any],
        device: torch.device,
    ) -> torch.Tensor:
        # get handler from log
        handler = logging.StreamHandler()
        handler.setFormatter(self.formatter)
        # get guid from input request
        log.debug(f"GuidLogFilter: {data}")
        guid = data.get(self.input_key)
        # Add filter to handler
        handler.addFilter(GuidLogFilter(guid=guid))
        # Get the root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        # Ensure propagation is enabled
        root_logger.propagate = True
        return data


class ResetLogging(Callable):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __call__(
        self,
        data: typing.Mapping[str, typing.Any],
        device: torch.device,
    ) -> torch.Tensor:
        # reset guid from input request
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s [%(guid)s]: - %(message)s"
            )
        )
        # handler.addFilter(logging.Filter(''))
        handler.addFilter(GuidLogFilter(guid=""))
        # Get the root logger and remove existing handlers
        root_logger = logging.getLogger()
        for hdlr in root_logger.handlers[:]:
            root_logger.removeHandler(hdlr)

        # Add the new handler to the root logger
        root_logger.addHandler(handler)
        root_logger.propagate = True

        return [{"is_success": True, "message": "Guid reset successfully."}]
