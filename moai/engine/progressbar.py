import rich.progress
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

__all__ = ["MoaiProgressBar"]

# progress_bar = RichProgressBar(
#     theme=RichProgressBarTheme(
#         description="green_yellow",
#         progress_bar="green1",
#         progress_bar_finished="green1",
#         progress_bar_pulse="#6206E0",
#         batch_progress="green_yellow",
#         time="grey82",
#         processing_speed="grey82",
#         metrics="grey82",
#         metrics_text_delimiter="\n",
#         metrics_format=".3e",
#     )


# NOTE: check https://github.com/Textualize/rich/discussions/482
# NOTE: check https://github.com/facebookresearch/EGG/blob/a139946a73d45553360a7f897626d1ae20759f12/egg/core/callbacks.py#L335
# NOTE: check https://github.com/Textualize/rich/discussions/921
class MoaiProgressBar(RichProgressBar):
    def __init__(self) -> None:
        super().__init__(
            theme=RichProgressBarTheme(metrics_text_delimiter="|"),
        )

    # return [
    #         TextColumn("[progress.description]{task.description}"),
    #         CustomBarColumn(
    #             complete_style=self.theme.progress_bar,
    #             finished_style=self.theme.progress_bar_finished,
    #             pulse_style=self.theme.progress_bar_pulse,
    #         ),
    #         BatchesProcessedColumn(style=self.theme.batch_progress),
    #         CustomTimeColumn(style=self.theme.time),
    #         ProcessingSpeedColumn(style=self.theme.processing_speed),
    #     ]
    def configure_columns(self, trainer: Trainer) -> list:
        original = super().configure_columns(trainer)
        moai_column = rich.progress.TextColumn(":moai:")
        spinner_column = rich.progress.SpinnerColumn(spinner_name="dots5")
        return [moai_column, spinner_column] + original
