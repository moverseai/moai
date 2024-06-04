import logging

import torch

log = logging.getLogger(__name__)

# NOTE: see this https://medium.com/@me_26124/debugging-neural-networks-6fa65742efd

__all__ = ["AnomalyDetection"]


class AnomalyDetection(object):
    def __init__(
        self,
    ):
        log.info(f"Toggling PyTorch anomaly detection ON.")
        # torch.set_anomaly_enabled()
        torch.autograd.set_detect_anomaly(True)
