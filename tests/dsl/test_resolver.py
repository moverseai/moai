from moai.core.execution.expression import TreeModule

import pytest
import torch
import lark

@pytest.mark.dsl
class TestDSL:
    def _parse_and_run(self, parser: lark.Lark, expression: str, tensors) -> torch.Tensor:
        tree = parser.parse(expression)
        m = TreeModule('check', tree)        
        m(tensors)
        return tensors['check']

    def test_combined(self, parser, various_tensors):
        expression = "test + ( another.number + add.this - 5) + stack(another.number2, test2, add.this2, 0) + cat(another.number3, test3, add.this3, 0)"
        x = self._parse_and_run(parser, expression, various_tensors)        
        y = torch.tensor([7.0, 9.0, 35.])        
        assert torch.equal(x, y)

    def test_add(self, parser, various_tensors):
        expression = "test2 + add.this2"
        x = self._parse_and_run(parser, expression, various_tensors)        
        y = torch.scalar_tensor(17.0)
        assert torch.equal(x, y)
    
    def test_adds(self, parser, various_tensors):
        expression = "test2 + add.this2 + (another.number2 + another.number3 + add.this3)"
        x = self._parse_and_run(parser, expression, various_tensors)        
        y = torch.scalar_tensor(34.0)
        assert torch.equal(x.squeeze(), y)

    def test_many_adds(self, parser, various_tensors):
        expression = "test2 + add.this2 + (another.number2 + another.number3 + add.this3) + (test2 + add.this2 + (test2 + add.this2))"
        x = self._parse_and_run(parser, expression, various_tensors)        
        y = torch.scalar_tensor(68.0)
        assert torch.equal(x.squeeze(), y)