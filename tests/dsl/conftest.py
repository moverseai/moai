from hydra_plugins.moai_dsl_plugin.moai_dsl_plugin import __MOAI_GRAMMAR__
from lark import Lark

import pytest
import torch
import numpy as np

@pytest.fixture
def parser():
    return Lark(__MOAI_GRAMMAR__, parser='earley', start='expr')

@pytest.fixture
def scalar_tensors():
    return {
        'test': 1, 
        'test2': torch.scalar_tensor(2),
        'test3': torch.scalar_tensor(2)[np.newaxis],
        'another': {
            'number': 10,
            'number2': torch.scalar_tensor(1),
            'number3': torch.scalar_tensor(1)[np.newaxis],
        }, 
        'add': {
            'this': -1,
            'this2': torch.scalar_tensor(15),
            'this3': torch.scalar_tensor(15)[np.newaxis],
        }
    }

@pytest.fixture
def shaped_tensors():
    return {
        'test': torch.tensor([[[1] *5]]),         
    }