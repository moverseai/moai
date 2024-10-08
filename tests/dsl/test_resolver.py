import lark
import numpy as np
import pytest
import torch

from moai.core.execution.expression import TreeModule


@pytest.mark.dsl
class TestDSL:
    def _parse_and_run(
        self, parser: lark.Lark, expression: str, tensors
    ) -> torch.Tensor:
        tree = parser.parse(expression)
        m = TreeModule("check", tree)
        return m(tensors)
        # return tensors['check']#TODO: revisit return name

    def _parse_and_run_dummy(
        self, parser: lark.Lark, expression: str, tensors
    ) -> torch.Tensor:
        tree = parser.parse(expression)
        m = TreeModule("check", tree)

    def test_combined(self, parser, scalar_tensors):
        expression = "test + ( another.number + add.this - 5) + stack(another.number2, test2, add.this2, 0) + cat(another.number3, test3, add.this3, 0)"
        x = self._parse_and_run(parser, expression, scalar_tensors)
        y = torch.tensor([7.0, 9.0, 35.0])
        expression = "test + ( another.number2 + add.this - 5) + stack(another.number3, test3, add.this3, 0) + cat(another.number3, test3, add.this3, 0)"
        x = self._parse_and_run(parser, expression, scalar_tensors)
        y = torch.tensor([[-2.0, -1.0, 12.0], [-1.0, 0.0, 13.0], [12.0, 13.0, 26.0]])
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
        expression = (
            "test2 + add.this2 + (another.number2 + another.number3 + add.this3)"
        )
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
        y = torch.scalar_tensor(2 / 15).double()
        assert torch.allclose(x, y)
        expression = "add.this2/test2"
        x = self._parse_and_run(parser, expression, scalar_tensors)
        y = torch.scalar_tensor(15 / 2).double()
        assert torch.allclose(x, y)

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
        expression = "5 + view(test, 5, 1) + five"
        x = self._parse_and_run(parser, expression, shaped_tensors)
        y = torch.tensor([11] * 5)
        assert torch.equal(x, y[:, np.newaxis])
        expression = "5 + view(test, 5, 1) * five"
        x = self._parse_and_run(parser, expression, shaped_tensors)
        y = torch.tensor([10] * 5)
        assert torch.equal(x, y[:, np.newaxis])
        expression = "(5 + view(test, 5, 1) ) * five"
        x = self._parse_and_run(parser, expression, shaped_tensors)
        y = torch.tensor([30] * 5)
        assert torch.equal(x, y[:, np.newaxis])

    def test_zeros(self, parser, shaped_tensors_cuda):
        expression = "onedim.threes * zeros(6)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        y = torch.zeros(6)[np.newaxis, :].to(x)
        assert torch.equal(x, y)
        expression = "zeros(6) * onedim.threes"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        y = torch.zeros(6)[np.newaxis, :].to(x)
        assert torch.equal(x, y)
        expression = "zeros(6) / onedim.threes"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        y = torch.zeros(6)[np.newaxis, :].to(x)
        assert torch.equal(x, y)

    def test_neg(self, parser, scalar_tensors):
        expression = "-test2"
        x = self._parse_and_run(parser, expression, scalar_tensors)
        y = torch.scalar_tensor(-2.0)
        assert torch.equal(x, y)

    def test_neg_expression(self, parser, scalar_tensors):
        expression = "-(test2 + test)"
        x = self._parse_and_run(parser, expression, scalar_tensors)
        y = torch.scalar_tensor(-3.0)
        assert torch.equal(x, y)

    def test_zeros_like(self, parser, scalar_tensors, shaped_tensors):
        expression = "zeros(test2)"
        x = self._parse_and_run(parser, expression, scalar_tensors)
        assert x.sum() == 0
        expression = "zeros(onedim.threes)"
        x = self._parse_and_run(parser, expression, shaped_tensors)
        assert x.sum() == 0

    def test_ones_like(self, parser, scalar_tensors, shaped_tensors):
        expression = "ones(test2)"
        x = self._parse_and_run(parser, expression, scalar_tensors)
        assert x.sum() == 1
        expression = "ones(onedim.threes)"
        x = self._parse_and_run(parser, expression, shaped_tensors)
        assert x.sum() == 6

    def test_rand_like(self, parser, scalar_tensors, shaped_tensors):
        expression = "rand(test3)"
        x = self._parse_and_run(parser, expression, scalar_tensors)
        assert True
        expression = "rand(onedim.threes)"
        x = self._parse_and_run(parser, expression, shaped_tensors)
        assert True

    def test_randn_like(self, parser, scalar_tensors, shaped_tensors):
        expression = "randn(test2)"
        x = self._parse_and_run(parser, expression, scalar_tensors)
        assert True
        expression = "randn(onedim.threes)"
        x = self._parse_and_run(parser, expression, shaped_tensors)
        assert True
        expression = "randn(random)"
        x = self._parse_and_run(parser, expression, shaped_tensors)
        assert x.shape == shaped_tensors["random"].shape

    def test_ones(self, parser, shaped_tensors_cuda):
        expression = "ones(6) * onedim.threes"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        y = torch.ones(6)[np.newaxis, :].to(x) * 3
        assert torch.equal(x, y)
        expression = "onedim.threes * ones(6)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        y = torch.ones(6)[np.newaxis, :].to(x) * 3
        assert torch.equal(x, y)
        expression = "onedim.threes / ones(6)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        y = torch.ones(6)[np.newaxis, :].to(x) * 3
        assert torch.equal(x, y)
        expression = "ones(6) / onedim.threes"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        y = torch.ones(6)[np.newaxis, :].to(x) / 3
        assert torch.equal(x, y)

    def test_sum_generated(self, parser, shaped_tensors_cuda):
        expression = "ones(6) + onedim.threes"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        y = torch.ones(6)[np.newaxis, :].to(x) + 3
        assert torch.equal(x, y)
        expression = "onedim.threes - ones(6)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        y = -torch.ones(6)[np.newaxis, :].to(x) + 3
        assert torch.equal(x, y)
        expression = "onedim.threes + (five * ones(6))"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        y = torch.ones(6)[np.newaxis, :].to(x) * 8
        assert torch.equal(x, y)
        expression = "(ones(onedim.threes) * 3) / onedim.threes"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        y = torch.ones(6)[np.newaxis, :].to(x)
        assert torch.equal(x, y)

    def test_squeeze(self, parser, shaped_tensors_cuda):
        expression = "unsq(onedim.threes, 0)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        assert len(x.shape) == 3 and x.shape[0] == 1
        expression = "sq(onedim.threes, 0)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        assert len(x.shape) == 1
        expression = "sq(onedim.threes)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        assert len(x.shape) == 1
        expression = "sq(fourdims, 0, 0)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        assert len(x.shape) == 2
        expression = "sq(fourdims, 0, 0, 0)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        assert len(x.shape) == 1
        expression = "sq(fourdims, 0, 0, 0, 0)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        assert len(x.shape) == 0
        # expression = "unsq(unsq(onedim.threes, 0), 0)"
        # x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        # assert len(x.shape) == 4 and x.shape[0] == 1
        expression = "unsq(onedim.threes, 0, 1)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        assert len(x.shape) == 4 and x.shape[0] == 1
        expression = "five * unsq(onedim.threes, 0, 1)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        assert len(x.shape) == 4 and x.shape[0] == 1 and x.ravel()[0] == 15.0
        expression = "five * unsq(test, 0, 1)"
        x = self._parse_and_run(parser, expression, shaped_tensors_cuda)
        assert len(x.shape) == 5 and x.shape[0] == 1 and x.ravel()[0] == 5.0

    def test_nested_access(self, parser, nested_tensors):
        expression = "single"
        x = self._parse_and_run(parser, expression, nested_tensors)
        assert float(x) == 1
        expression = "outer.five"
        x = self._parse_and_run(parser, expression, nested_tensors)
        assert float(x) == 5
        expression = "outer.inner.five_ones"
        x = self._parse_and_run(parser, expression, nested_tensors)
        assert len(x.shape) == 3 and torch.sum(x) == 5

    def test_slicing(self, parser, highdim_tensors):
        expression = "single[..., [2,5,6,7], -1, 2:5, :]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert x.sum() == 360.0
        expression = "single[..., [2,5,6,7], -1, 2:5, :] + test"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert x.sum() == 720.0
        expression = "single[..., [2,5,6,7], -1, 2:5, :] + ones(rand)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert x.sum() == 720.0
        expression = "single[..., [2,5,6,7], -1, 2:5, :] + ones(10, 4, 3, 3)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert x.sum() == 720.0
        expression = "multi[..., 2:5, :]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert x.sum() == 80.0
        expression = "multi[..., :, 2:5]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert x.sum() == 75.0
        expression = "multi[:, 2:5]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert x.sum() == 75.0
        expression = "linspace[2:5]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.allclose(x, torch.linspace(0, 10, 10)[2:5])
        expression = "linspace3[:, 2:5, :]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.allclose(x.squeeze(), torch.linspace(0, 10, 10)[2:5])
        expression = "linspace3[..., 2:5, 0]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.allclose(x.squeeze(), torch.linspace(0, 10, 10)[2:5])
        expression = "linspace3[..., -1, 0]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.allclose(x.squeeze(), torch.scalar_tensor(10))
        expression = "linspace3[-1, -1, -1]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.allclose(x.squeeze(), torch.scalar_tensor(10))
        expression = "linspace3[:, 9, :]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.allclose(x.squeeze(), torch.scalar_tensor(10))
        expression = "multi[1:3, 2:5]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        y = torch.tensor([[3, 4, 5], [0, 0, 0]])
        assert torch.allclose(x, y)
        # NOTE: this is not yet supported (implied start/end)
        # expression = "multi[:3, 2:5]"
        # x = self._parse_and_run(parser, expression, highdim_tensors)
        # y = torch.tensor([[3, 4, 5], [0, 0, 0]])
        # assert torch.allclose(x, y)

    def test_transpose(self, parser, highdim_tensors):
        expression = "transpose(fourdim, 1, 0)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        torch.equal(x, highdim_tensors["fourdim"].transpose(1, 0))
        expression = "transpose(fourdim, 2, 1, 3, 0)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        torch.equal(x, highdim_tensors["fourdim"].permute(2, 1, 3, 0))

    def test_exp_log(self, parser, highdim_tensors):
        expression = "exp(fourdim)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(x, highdim_tensors["fourdim"].exp())
        expression = "log(fourdim)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(x, highdim_tensors["fourdim"].log())
        expression = "log(fourdim) + exp(fourdim)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(
            x, highdim_tensors["fourdim"].log() + highdim_tensors["fourdim"].exp()
        )
        expression = "log(fourdim + 1) + exp(fourdim - 2)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(
            x,
            (highdim_tensors["fourdim"] + 1).log()
            + (highdim_tensors["fourdim"] - 2).exp(),
        )
        expression = "log(exp(fourdim))"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(x, highdim_tensors["fourdim"].exp().log())
        expression = "log(exp(fourdim)) + single[:5, :3, :, :, 0]"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(
            x,
            highdim_tensors["fourdim"].exp().log()
            + highdim_tensors["single"][:5, :3, :, :, 0],
        )

    def test_reciprocal(self, parser, highdim_tensors):
        expression = "reciprocal(fourdim)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(x, highdim_tensors["fourdim"].reciprocal())
        expression = "reciprocal(reciprocal(fourdim))"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(x, x)

    def test_flatten(self, parser, highdim_tensors):
        expression = "flatten(fourdim, 1)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(x, highdim_tensors["fourdim"].flatten(1))
        expression = "flatten(fourdim, 0, 3)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(x, highdim_tensors["fourdim"].flatten(0, 3))
        expression = "flatten(fourdim, 0) + ones(180)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert x.sum() == 300.0
        expression = "flatten(fourdim, 1) + ones(5, 36)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert x.sum() == 300.0

    def test_repeat_interleave(self, parser, highdim_tensors):
        expression = "repeat_interleave(fourdim, 2, 1)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(x, highdim_tensors["fourdim"].repeat_interleave(2, 1))
        expression = "repeat_interleave(fourdim, 2, 0)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert torch.equal(x, highdim_tensors["fourdim"].repeat_interleave(2, 0))
        expression = "repeat_interleave(single, 2, 1)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert x.sum() == 3600 * 2  # 3600 is the sum of single and 2 is the repeat
        expression = "repeat_interleave(fourdim, 2, 0)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert (x[5:] != highdim_tensors["fourdim"]).sum() == 0
        expression = "repeat_interleave(fourdim, 2, 0) + ones(10, 3, 2, 6)"
        x = self._parse_and_run(parser, expression, highdim_tensors)
        assert x.sum() == 120 * 2 + 10 * 3 * 2 * 6

    def test_trig(self, parser, trig_tensors):
        expression = "sin(pi2)"
        x = self._parse_and_run(parser, expression, trig_tensors)
        assert torch.allclose(x, torch.sin(torch.scalar_tensor(np.pi * 0.5)))
        expression = "asin(rand)"
        x = self._parse_and_run(parser, expression, trig_tensors)
        assert torch.allclose(x, trig_tensors["rand"].asin())
        expression = "cos(pi2)"
        x = self._parse_and_run(parser, expression, trig_tensors)
        assert torch.allclose(x, torch.cos(torch.scalar_tensor(np.pi * 0.5)))
        expression = "acos(rand)"
        x = self._parse_and_run(parser, expression, trig_tensors)
        assert torch.allclose(x, trig_tensors["rand"].acos())
        expression = "tan(pi2)"
        x = self._parse_and_run(parser, expression, trig_tensors)
        assert torch.allclose(x, torch.tan(torch.scalar_tensor(np.pi * 0.5)))
        expression = "atan(rand)"
        x = self._parse_and_run(parser, expression, trig_tensors)
        assert torch.allclose(x, trig_tensors["rand"].atan())
        expression = "abs(mrand)"
        x = self._parse_and_run(parser, expression, trig_tensors)
        assert torch.allclose(x, torch.abs(trig_tensors["mrand"]))
        expression = "abs(mrand - 1)"
        x = self._parse_and_run(parser, expression, trig_tensors)
        assert torch.allclose(x, torch.abs(trig_tensors["mrand"] - 1))
        expression = "abs(rand - 1)"
        x = self._parse_and_run(parser, expression, trig_tensors)
        assert torch.allclose(x, torch.abs(trig_tensors["rand"] - 1))
        expression = "sin(pi2)^2 + cos(pi2)^2"
        x = self._parse_and_run(parser, expression, trig_tensors)
        assert x == 1
        expression = "sin(rand)^2 + cos(rand)^2"
        x = self._parse_and_run(parser, expression, trig_tensors)
        assert x.squeeze() == torch.scalar_tensor(1.0)

    def test_rad_deg(self, parser, trig_tensors):
        expression = "deg(pi2)"
        x = self._parse_and_run(parser, expression, trig_tensors)
        trig_tensors["x"] = x
        expression = "rad(x)"
        y = self._parse_and_run(parser, expression, trig_tensors)
        assert y == np.pi * 0.5

    def test_expand_batch_as(self, parser, varying_shape_tensors):
        expression = "expand_batch_as(twodim_b1, twodim)"
        x = self._parse_and_run(parser, expression, varying_shape_tensors)
        assert varying_shape_tensors["twodim"].shape[0] == x.shape[0]
