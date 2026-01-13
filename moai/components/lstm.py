import logging

import omegaconf.omegaconf
import torch

log = logging.getLogger(__name__)

__all__ = ["LSTM"]


class LSTM(torch.nn.Module):
    def __init__(
        self,
        configuration: omegaconf.DictConfig,
    ) -> None:
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=configuration.input_size,
            hidden_size=configuration.hidden_size,
            num_layers=configuration.num_layers,
            batch_first=configuration.batch_first,
            dropout=configuration.dropout,
            bidirectional=configuration.bidirectional,
        )
        self.init_hidden_state = torch.nn.Parameter(
            torch.zeros(
                configuration.num_layers * (2 if configuration.bidirectional else 1),
                configuration.batch_size,
                configuration.hidden_size,
            )
        )
        self.init_cell_state = torch.nn.Parameter(
            torch.zeros(
                configuration.num_layers * (2 if configuration.bidirectional else 1),
                configuration.batch_size,
                configuration.hidden_size,
            )
        )

    def forward(self, input: torch.Tensor):
        return self.lstm(input, (self.init_hidden_state, self.init_cell_state))
