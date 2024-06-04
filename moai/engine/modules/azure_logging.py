import logging

log = logging.getLogger(__name__)

# NOTE: https://learn.microsoft.com/en-us/azure/developer/python/sdk/azure-sdk-logging


class AzureLogging(object):
    def __init__(
        self,
        level: str = "INFO",
    ):
        """
        Sets up logging level to Azure Blob Storage.
        """
        loggers = ["azure"]
        for lg in loggers:
            log.info(f"Logging {lg} logger's level set to {level}.")
            logging.getLogger(lg).setLevel(level)
            log.debug(f"Logger {logging.getLogger(lg)}.")
