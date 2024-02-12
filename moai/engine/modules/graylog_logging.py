# get the logger instance and change the config
# Assuming graylog.utc_formatter.UTCFormatter and other necessary modules are already defined
import logging.config
from graylog import UTCFormatter
import colorlog
import os


class GraylogLogging(object):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 12201,
        level: str = "INFO",
    ):
        """
        Sets up logging level to Graylog.
        """
        # define the logging configuration

        LOGGING_CONFIG = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "utc": {
                    "()": UTCFormatter,
                    "format": "[%(asctime)s UTC][%(name)s][%(levelname)s] - %(message)s",
                },
                "colorlog": {
                    "()": colorlog.ColoredFormatter,
                    "format": "[%(cyan)s%(asctime)s UTC%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
                    "log_colors": {
                        "DEBUG": "purple",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red",
                    },
                },
                "simple": {
                    "format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
                },
            },
            "handlers": {
                "graylog": {
                    "class": "graypy.GELFUDPHandler",
                    "formatter": "utc",
                    "host": host,
                    "port": port,
                },
                "file": {
                    "class": "logging.FileHandler",
                    "formatter": "simple",
                    "filename": f"{os.environ.get('HYDRA_JOB_NAME', 'default_job')}.log",
                },
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "colorlog",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": os.environ.get("HYDRA_GRAYLOG_LOG_LEVEL", "INFO"),
                "handlers": ["graylog", "console", "file"],
            },
        }

        # Applying the logging configuration
        logging.config.dictConfig(LOGGING_CONFIG)
        # set the logging level
        logging.getLogger().setLevel(level)
