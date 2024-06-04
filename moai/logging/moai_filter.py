import logging


class MoaiFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__("moai")

    def filter(self, record):
        record.msg = f":moai: {record.msg}"
        return record
