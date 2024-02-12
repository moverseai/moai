from moai.engine.modules.seed import ManualSeed
from moai.engine.modules.anomaly_detection import AnomalyDetection
from moai.engine.modules.nvidia import (
    NvidiaSMI,
    DebugCUDA,
    DisableCuDNN,
    BlockingCUDA,
    DisableTensorCores,
    DisableFP16Matmul,
)
from moai.engine.modules.importer import Import
from moai.engine.modules.empty_cache import EmptyCache
from moai.engine.modules.azure_logging import AzureLogging

__all__ = [
    "AnomalyDetection",
    "DebugCUDA",
    "EmptyCache",
    "Import",
    "NvidiaSMI",
    "Seed",
    'DisableCuDNN',
    'BlockingCUDA',
    'DisableTensorCores',
    'DisableFP16Matmul',
    'AzureLogging',
]

try:
    from moai.engine.modules.graylog_logging import GraylogLogging    
    __all__.append('GraylogLogging')
    from moai.engine.modules.hydra_logging import HydraLogging
except Exception:
    pass #TODO: log this

