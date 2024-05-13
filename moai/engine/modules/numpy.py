import numpy as np
import logging
from packaging.version import Version

__all__ = [
    'Numpy',
]

log = logging.getLogger(__name__)

class Numpy(object):
    def __init__(self):
        if Version(np.__version__) > Version("1.23.1"):
            log.info("Numpy version > 1.23.1, adding deprecated types.")
            np.bool = np.bool_
            np.int = np.int_
            np.float = np.float_
            np.complex = np.complex_
            np.object = np.object_
            np.unicode = np.unicode_
            np.str = np.str_
