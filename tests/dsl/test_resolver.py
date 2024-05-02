from moai.core.execution.expression import TreeModule

import pytest
import torch
import lark
import numpy as np

@pytest.mark.dsl
class TestDSL:
    def _parse_and_run(self, parser: lark.Lark, expression: str, tensors) -> torch.Tensor:
        tree = parser.parse(expression)
        m = TreeModule('check', tree)        
        m(tensors)
        return tensors['check']

    def test_combined(self, parser, scalar_tensors):
        expression = "test + ( another.number + add.this - 5) + stack(another.number2, test2, add.this2, 0) + cat(another.number3, test3, add.this3, 0)"
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.tensor([7.0, 9.0, 35.])        
        assert torch.equal(x, y)

    def test_add(self, parser, scalar_tensors):
        expression = "test2 + add.this2"
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.scalar_tensor(17.0)
        assert torch.equal(x, y)
        expression = "add.this2+test2"
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.scalar_tensor(17.0)
        assert torch.equal(x, y)
    
    def test_adds(self, parser, scalar_tensors):
        expression = "test2 + add.this2 + (another.number2 + another.number3 + add.this3)"
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.scalar_tensor(34.0)
        assert torch.equal(x.squeeze(), y)

    def test_many_adds(self, parser, scalar_tensors):
        expression = "test2 + add.this2 + (another.number2 + another.number3 + add.this3) + (test2 + add.this2 + (test2 + add.this2))"
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.scalar_tensor(68.0)
        assert torch.equal(x.squeeze(), y)
        expression = "(test2 + add.this2 + another.number2) + another.number3 + ((add.this3 + test2) + add.this2 + (test2 + add.this2))"
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.scalar_tensor(68.0)
        assert torch.equal(x.squeeze(), y)

    def test_mul(self, parser, scalar_tensors):
        expression = "test2 * add.this2"
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.scalar_tensor(30.0)
        assert torch.equal(x, y)
        expression = "add.this2*test2"
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.scalar_tensor(30.0)
        assert torch.equal(x, y)

    def test_div(self, parser, scalar_tensors):
        expression = "test2 / add.this2"
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.scalar_tensor(2/15)
        assert torch.equal(x, y)
        expression = "add.this2/test2"
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.scalar_tensor(15/2)
        assert torch.equal(x, y)

    def test_pow(self, parser, scalar_tensors):
        expression = "add.this2 ^ test2"
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.scalar_tensor(15**2)
        assert torch.equal(x, y)
        expression = "test2 ^ add.this2 "
        x = self._parse_and_run(parser, expression, scalar_tensors)        
        y = torch.scalar_tensor(2**15)
        assert torch.equal(x, y)

    def test_reshape(self, parser, shaped_tensors):
        expression = "view(test, 5, 1)"
        x = self._parse_and_run(parser, expression, shaped_tensors)        
        y = (5, 1)
        assert np.all(np.equal(x.shape, y))