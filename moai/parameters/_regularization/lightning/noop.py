import pytorch_lightning.callbacks as pytlc

__all__ = ["NoOp"]

class NoOp(pytlc.Callback):
    def __init__(self, **kwargs):
        super(NoOp, self).__init__(
            **kwargs
        )