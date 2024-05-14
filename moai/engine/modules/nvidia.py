import os
import torch
import logging

__all__ = [
    'NvidiaSMI',
    'DebugCUDA',
    'DisableCuDNN',
    'BlockingCUDA',
    'DisableTensorCores',
    'DisableFP16Matmul',
]

log = logging.getLogger(__name__)

class NvidiaSMI(object):
    def __init__(self):
        if 'PROGRAMFILES' in os.environ.keys():
            nvidia_smi_path = os.path.join(
                os.environ['PROGRAMFILES'],
                'NVIDIA Corporation',
                'NVSMI'
            )
            if nvidia_smi_path not in os.environ['PATH']:
                os.environ['PATH'] = os.environ['PATH'] + ";" + nvidia_smi_path

class DebugCUDA(object):
    def __init__(self):
        os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)

class DisableCuDNN(object):
    def __init__(self):
        log.info("Disabling CuDNN.")
        torch.backends.cudnn.enabled = False

class BlockingCUDA(object):
    def __init__(self):
        log.info("Setting CUDA to blocking execution mode.")
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class DisableTensorCores(object):
    def __init__(self):
        log.info("Disabling Tensor Cores.")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

class DisableFP16Matmul(object):
    def __init__(self):
        log.info("Disabling FP16 Matmul.")
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False