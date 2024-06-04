import logging
from datetime import datetime

from pytz import timezone


class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, timezone("UTC"))
        return dt.strftime(datefmt if datefmt else "%Y-%m-%d %H:%M:%S")
