import os
import pytorch_lightning

class NoOp(pytorch_lightning.callbacks.ModelCheckpoint):
    def __init__(self, **kwargs):
        super(NoOp, self).__init__(
            filepath=os.getcwd(),
            monitor="",
            verbose=False,
            save_top_k=0,
            period=0,
            save_weights_only=False,
            mode='auto'
        )

    def _save_model(self, filepath: str):
        pass

    def on_validation_end(self, trainer, pl_module):
        pass