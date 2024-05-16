from hydra_plugins.moai_dsl_plugin.moai_dsl_plugin import __MOAI_GRAMMAR__
from lark import Lark

import pytest
import torch
import numpy as np
import benedict

@pytest.fixture
def parser():
    return Lark(__MOAI_GRAMMAR__, parser='earley', start='expr')

@pytest.fixture
def scalar_tensors():
    return benedict.benedict({
        'test': 1, 
        'test2': torch.scalar_tensor(2).double(),
        'test3': torch.scalar_tensor(2)[np.newaxis].float(),
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
    })

@pytest.fixture
def shaped_tensors():
    return benedict.benedict({
        'test': torch.tensor([[[1] *5]]),
        'five': torch.scalar_tensor(5.0),
        'onedim': {
            'threes': torch.tensor([[3] * 6]).float()
        }
    })

@pytest.fixture
def shaped_tensors_cuda():
    return benedict.benedict({
        'test': torch.tensor([[[1] *5]]).cuda(),
        'five': torch.scalar_tensor(5).cuda(),
        'onedim': {
            'threes': torch.tensor([[3] * 6]).cuda().double()
        }
    })

@pytest.fixture
def nested_tensors():
    return benedict.benedict({
        'single': torch.scalar_tensor(1.0),
        'outer': {
            'inner': {
                'five_ones': torch.tensor([[[1] * 5]]),
            },
            'five': torch.scalar_tensor(5),
        }
    })

@pytest.fixture
def highdim_tensors():
    ones = torch.ones(5, 3, 2, 6)
    ones[:, 0] = 0.0    
    return benedict.benedict({
        'single': torch.ones(10, 10, 2, 6, 3),
        'test': torch.ones(10, 4, 3, 3),
        'rand': torch.rand(10, 4, 3, 3),
        'multi': torch.tensor([
            [5, 5, 5, 5, 5],
            [1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [15, 15, 15, 15, 15],
        ]),
        'linspace': torch.linspace(0, 10, 10),
        'linspace3': torch.linspace(0, 10, 10)[np.newaxis, :, np.newaxis],
        'fourdim': ones,
    })