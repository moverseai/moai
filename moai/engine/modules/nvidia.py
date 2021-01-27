import os

__all__ = ['NvidiaSMI', 'DebugCUDA']

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