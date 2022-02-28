import moai.utils.engine as mieng

import omegaconf.omegaconf

__all__ = ['Statistics']

class Statistics(mieng.Collection):
    def __init__(self, 
        calculators: omegaconf.DictConfig,        
    ):
        super().__init__(items=calculators, name="calculators")