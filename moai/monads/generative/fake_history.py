import random

import torch

__all__ = ["FakeHistory"]


class FakeHistory(torch.nn.Module):
    def __init__(
        self,
        history_size: int,
        percentage: float = 0.5,
    ) -> None:
        super().__init__()
        self.history_size = history_size
        self.percentage = percentage
        # NOTE: we are not using buffers to store the history
        #       as we do not need them to persist
        #       and in order to do that we need the tensor size as input
        self.actual_size = 0
        self.history = []

    def forward(self, fake: torch.Tensor) -> torch.Tensor:
        out = []
        b = fake.shape[0]
        for i in range(b):
            if self.actual_size < self.history_size:
                self.history.append(fake[i].detach().clone())
                out.append(fake[i])
                self.actual_size += 1
            else:
                p = random.uniform(0, 1)
                if p < self.percentage:
                    id = random.randint(1, self.history_size) - 1
                    tmp = self.history[id]
                    self.history[id] = fake[i].detach().clone()
                    out.append(tmp)
                else:
                    out.append(fake[i])
        return torch.stack(out)


class GradualFakeHistory(torch.nn.Module):
    def __init__(
        self,
        history_size: int,
        percentage: float = 0.5,
    ) -> None:
        super().__init__()
        self.history_size = history_size
        self.percentage = percentage
        # NOTE: we are not using buffers to store the history
        #       as we do not need them to persist
        #       and in order to do that we need the tensor size as input
        self.actual_size = 0
        self.history = []

    def forward(self, fake: torch.Tensor) -> torch.Tensor:
        out = []
        b = fake.shape[0]
        if (len(self.history) > 0) and (self.history[0].shape[-1] != fake.shape[-1]):
            self.history = []
            self.actual_size = 0
        for i in range(b):
            if self.actual_size < self.history_size:
                self.history.append(fake[i].detach().clone())
                out.append(fake[i])
                self.actual_size += 1
            else:
                p = random.uniform(0, 1)
                if p < self.percentage:
                    id = random.randint(1, self.history_size) - 1
                    tmp = self.history[id]
                    self.history[id] = fake[i].detach().clone()
                    out.append(tmp)
                else:
                    out.append(fake[i])
        return torch.stack(out)
