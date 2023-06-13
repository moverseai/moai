from moai.engine.modules.clearml import _get_logger
from moai.engine.modules.clearml import _get_project_name
from moai.engine.modules.clearml import _get_task_name


import numpy as np
import pytorch_lightning
import logging
import typing
import toolz
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["ClearML"]


class ClearML(pytorch_lightning.loggers.base.LightningLoggerBase):
    def __init__(self):
        super(ClearML, self).__init__()
        self._name, self._version = _get_project_name(), _get_task_name()
        self.logger = _get_logger()

    @pytorch_lightning.loggers.base.rank_zero_only
    def log_metrics(self, metrics: typing.Dict[str, typing.Any], step: int) -> None:
        dataloader_index = toolz.get_in(
            ["dataloader_index"],
            toolz.keyfilter(lambda k: k.startswith("__moai__"), metrics).popitem()[1],
        ) if toolz.keyfilter(lambda k: k.startswith("__moai__"), metrics) else None
        if dataloader_index is not None:
            metrics = toolz.keyfilter(
                lambda k: k.endswith(str(int(dataloader_index))),
                metrics,
            )
        train_metrics = toolz.keymap(
            lambda k: k.replace("train_", ""),
            toolz.keyfilter(lambda k: k.startswith("train_"), metrics),
        )
        val_metrics = toolz.keymap(
            lambda k: k.replace("val_", ""),
            toolz.keyfilter(lambda k: k.startswith("val_"), metrics),
        )
        test_metrics = toolz.keymap(
            lambda k: k.replace("test_", "").replace("/epoch_0", ""),
            toolz.keyfilter(lambda k: k.startswith("test_"), metrics),
        )
        # test_metrics = toolz.keymap(lambda k: k.replace('test_', '').replace('/epoch_0', ''),
        #     toolz.keyfilter(
        #         lambda k: int(k.split('/')[2].split("_")[-1]) == int(dataloader_index),
        #         (toolz.keyfilter(lambda k: k.startswith('test_'), metrics))
        #     ) if dataloader_index is not None else toolz.keyfilter(lambda k: k.startswith('test_'), metrics)
        # )
        if train_metrics:
            loss = float(metrics["total_loss"])
            self.logger.report_scalar("train", "loss", loss, step)
            for k, v in train_metrics.items():
                self.logger.report_scalar("train", k, v, step)
        elif test_metrics:
            # return #TODO: test case
            dataset_test_metrics = toolz.valmap(
                lambda v: toolz.keymap(lambda k: k.split("/")[0], dict(v)),
                toolz.groupby(
                    lambda k: toolz.get(1, k[0].split("/"), "metrics"),
                    test_metrics.items(),
                ),
            )
            for d, m in dataset_test_metrics.items():
                for k, v in m.items():
                    self.logger.report_scalar(d, k, v, step)
                df = pd.DataFrame(dataset_test_metrics[d],
                    index=["sample"+str(step)],
                )
                df.index.name = "id"
                self.logger.report_table(
                    "Metrics", 
                    "Per Sample", 
                    iteration=step, 
                    table_plot=df
                )
        if val_metrics:
            e = int(metrics["epoch"])
            dataset_val_metrics = toolz.valmap(
                lambda v: toolz.keymap(lambda k: k.split("/")[0], dict(v)),
                toolz.groupby(
                    lambda k: toolz.get(1, k[0].split("/"), "metrics"),
                    val_metrics.items(),
                ),
            )
            for d, m in dataset_val_metrics.items():
                for k, v in m.items():
                    self.logger.report_scalar(d, k, v, e)

    @pytorch_lightning.loggers.base.rank_zero_only
    def log_hyperparams(
        self, params: typing.Dict[str, typing.Any]  # TODO: or namespace object ?
    ) -> None:
        pass

    @pytorch_lightning.loggers.base.rank_zero_only
    def save(self) -> None:
        pass

    @pytorch_lightning.loggers.base.rank_zero_only
    def finalize(self, status: str) -> None:
        """Do any processing that is necessary to finalize an experiment
        :param status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        self.save()

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        return self._version

    @property
    def experiment(self) -> typing.Any:
        return self.name
