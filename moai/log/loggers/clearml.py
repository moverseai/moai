import logging
import typing

import numpy as np
import pandas as pd
import pytorch_lightning
import toolz
from pytorch_lightning.loggers.logger import DummyExperiment
from typing_extensions import override

from moai.engine.modules.clearml import _get_logger, _get_project_name, _get_task_name

log = logging.getLogger(__name__)

__all__ = ["ClearML"]


class ClearML(pytorch_lightning.loggers.Logger):
    def __init__(self):
        super(ClearML, self).__init__()
        self._name, self._version = _get_project_name(), _get_task_name()
        self.logger = _get_logger()

    @override
    def log_metrics(self, metrics: typing.Dict[str, typing.Any], step: int) -> None:
        train_metrics = toolz.keymap(
            lambda k: k.replace("train/", ""),
            toolz.keyfilter(lambda k: k.startswith("train/"), metrics),
        )
        val_metrics = toolz.keymap(
            lambda k: k.replace("val/", ""),
            toolz.keyfilter(lambda k: k.startswith("val/"), metrics),
        )
        test_metrics = toolz.keymap(
            lambda k: k.replace("test/", "").replace("/epoch_0", ""),
            toolz.keyfilter(lambda k: k.startswith("test/"), metrics),
        )
        # test_metrics = toolz.keymap(lambda k: k.replace('test_', '').replace('/epoch_0', ''),
        #     toolz.keyfilter(
        #         lambda k: int(k.split('/')[2].split("_")[-1]) == int(dataloader_index),
        #         (toolz.keyfilter(lambda k: k.startswith('test_'), metrics))
        #     ) if dataloader_index is not None else toolz.keyfilter(lambda k: k.startswith('test_'), metrics)
        # )
        if train_metrics:
            loss = float(metrics["train/loss/total"])
            self.logger.report_scalar("train", "loss", loss, step)
            for k, v in train_metrics.items():
                self.logger.report_scalar("train", k, v, step)
        elif test_metrics:
            # return #TODO: test case
            dataset_test_metrics = toolz.valmap(
                lambda v: toolz.keymap(lambda k: k.split("/")[1], dict(v)),
                toolz.groupby(
                    lambda k: toolz.get(2, k[0].split("/"), "metrics"),
                    test_metrics.items(),
                ),
            )
            for d, m in dataset_test_metrics.items():
                for k, v in m.items():
                    self.logger.report_scalar(d, k, v, step)
                df = pd.DataFrame(
                    dataset_test_metrics[d],
                    index=["sample" + str(step)],
                )
                df.index.name = "id"
                self.logger.report_table(
                    "Metrics", "Per Sample", iteration=step, table_plot=df
                )
        if val_metrics:
            # TODO: report the average of the metrics
            dataset_val_metrics = toolz.valmap(
                lambda v: toolz.keymap(lambda k: k.split("/")[1], dict(v)),
                toolz.groupby(
                    lambda k: toolz.get(2, k[0].split("/"), "metrics"),
                    val_metrics.items(),
                ),
            )
            # dataset = list(val_metrics.items())[0][0].split("/")[2]
            for dataset_name, dataset_metrics in dataset_val_metrics.items():
                for metric_name, metric_value in dataset_metrics.items():
                    self.logger.report_scalar(
                        dataset_name,
                        metric_name,
                        metric_value,
                        int(
                            toolz.keyfilter(
                                lambda k: k.startswith("epoch"), metrics
                            ).popitem()[1]
                        ),
                    )

    def log_hyperparams(
        self, params: typing.Dict[str, typing.Any]  # TODO: or namespace object ?
    ) -> None:
        my_params = self.logger.task.connect_configuration(my_params)

    def save(self) -> None:
        pass

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
    @pytorch_lightning.loggers.logger.rank_zero_experiment
    def experiment(self) -> "DummyExperiment":
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in your code, do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment
        assert (
            pytorch_lightning.loggers.logger.rank_zero_experiment.rank == 0
        ), "tried to init log dirs in non global_rank=0"

        return self._experiment

    # @property
    # def experiment(self) -> typing.Any:
    #     return self.name
