"""
PyTorch Optimizer wrapper for CUDA L-BFGS

This provides a drop-in replacement for torch.optim.LBFGS that uses
the CUDA-accelerated L-BFGS implementation. Compatible with PyTorch Lightning.
"""

# Standard Library
from typing import Callable, Optional

# Third Party
import torch
import torch.optim

try:
    # CuRobo
    from lbfgs_cuda import lbfgs_cuda

    CUDA_AVAILABLE = True
except ImportError:
    try:
        # CuRobo
        import lbfgs_step_cuda

        def lbfgs_cuda(
            step_vec,
            rho_buffer,
            y_buffer,
            s_buffer,
            q,
            grad_q,
            x_0,
            grad_0,
            epsilon=0.1,
            batch_size=None,
            m=None,
            v_dim=None,
            stable_mode=False,
            use_shared_buffers=True,
        ):
            if batch_size is None:
                batch_size = grad_q.shape[0]
            if m is None:
                m = y_buffer.shape[0]
            if v_dim is None:
                v_dim = grad_q.shape[1]
            result = lbfgs_step_cuda.lbfgs_step_cuda(
                step_vec,
                rho_buffer,
                y_buffer,
                s_buffer,
                q,
                grad_q,
                x_0,
                grad_0,
                epsilon,
                batch_size,
                m,
                v_dim,
                stable_mode,
                use_shared_buffers,
            )
            return result[0]

        CUDA_AVAILABLE = True
    except ImportError:
        CUDA_AVAILABLE = False
        lbfgs_cuda = None


class LBFGSCuda(torch.optim.Optimizer):
    """
    L-BFGS optimizer using CUDA-accelerated implementation.

    Drop-in replacement for torch.optim.LBFGS with CUDA acceleration.
    Optimized for batch optimization and high-dimensional problems.

    Compatible with PyTorch Lightning and standard PyTorch training loops.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1.0)
        max_iter: Maximal number of iterations per optimization step (default: 20)
        max_eval: Maximal number of function evaluations per optimization step
                  (default: max_iter * 1.25)
        tolerance_grad: Termination tolerance on first order optimality (default: 1e-5)
        tolerance_change: Termination tolerance on function value/parameter changes
                          (default: 1e-9)
        history_size: Update history size (default: 100)
        line_search_fn: Line search function ('strong_Wolfe' or None, default: None)
                        Note: Currently only None is supported (fixed step size)
        epsilon: Regularization parameter for CUDA kernel (default: 0.1)
        stable_mode: Use stable mode for CUDA kernel (default: False)
        use_shared_buffers: Use shared memory optimization (default: True)

    Example:
        >>> optimizer = LBFGSCuda(model.parameters(), lr=1.0, history_size=5)
        >>> def closure():
        ...     optimizer.zero_grad()
        ...     loss = loss_fn(model(input), target)
        ...     loss.backward()
        ...     return loss
        >>> optimizer.step(closure)

    Example with PyTorch Lightning:
        >>> from lbfgs_cuda_optimizer import LBFGSCuda
        >>>
        >>> class MyModel(pl.LightningModule):
        ...     def configure_optimizers(self):
        ...         return LBFGSCuda(self.parameters(), lr=1.0, history_size=5)
        ...
        ...     def training_step(self, batch, batch_idx):
        ...         # Lightning automatically handles closure for L-BFGS
        ...         ...
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-5,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: Optional[str] = None,
        epsilon: float = 0.1,
        stable_mode: bool = False,
        use_shared_buffers: bool = True,
    ):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA L-BFGS not available. Please build lbfgs_step_cuda extension first.\n"
                "Run: python setup_lbfgs_only.py build_ext --inplace"
            )

        if max_eval is None:
            max_eval = max_iter * 5 // 4

        if line_search_fn is not None and line_search_fn != "strong_Wolfe":
            raise ValueError(
                "Only 'strong_Wolfe' or None is supported for line_search_fn"
            )

        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
            epsilon=epsilon,
            stable_mode=stable_mode,
            use_shared_buffers=use_shared_buffers,
        )

        super(LBFGSCuda, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGSCuda doesn't support per-parameter options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

        # Check if all parameters are on CUDA
        for p in self._params:
            if not p.is_cuda:
                raise RuntimeError(
                    "LBFGSCuda requires all parameters to be on CUDA device. "
                    f"Found parameter on device: {p.device}"
                )

    def _numel(self):
        """Total number of parameters"""
        if self._numel_cache is None:
            self._numel_cache = sum(p.numel() for p in self._params)
        return self._numel_cache

    def _gather_flat_params(self):
        """Gather all parameters into a single flat tensor"""
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_grad(self):
        """Gather all gradients into a single flat tensor"""
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new_zeros(p.numel())
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        """Add update to parameters"""
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.add_(
                update[offset : offset + numel].view_as(p.data), alpha=step_size
            )
            offset += numel

    def _clone_param(self):
        """Clone current parameters"""
        return [p.clone() for p in self._params]

    def _set_param(self, params_data):
        """Set parameters from cloned data"""
        for p, pdata in zip(self._params, params_data):
            p.data.copy_(pdata)

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                    Required for L-BFGS optimizer.

        Returns:
            Loss value from closure
        """
        if closure is None:
            raise ValueError("LBFGSCuda requires a closure function")

        assert len(self.param_groups) == 1
        group = self.param_groups[0]

        lr = group["lr"]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        history_size = group["history_size"]
        epsilon = group["epsilon"]
        stable_mode = group["stable_mode"]
        use_shared_buffers = group["use_shared_buffers"]
        line_search_fn = group["line_search_fn"]

        # Get or initialize state
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # Evaluate initial f(x) and df/dx
        orig_loss = closure()

        if not torch.isfinite(orig_loss):
            raise RuntimeError(
                "The loss returned by the closure is not finite. "
                "This usually means that the model parameters are not "
                "initialized properly or the learning rate is too high."
            )

        loss = orig_loss.item()
        current_evals = 1
        state["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        if opt_cond:
            return orig_loss

        # Get state variables
        old_dirs = state.get("old_dirs", [])  # List of y (gradient differences)
        old_stps = state.get("old_stps", [])  # List of s (parameter differences)
        ro = state.get("ro", [])  # List of rho (1 / (y^T s))
        H_diag = state.get("H_diag", 1.0)
        prev_flat_grad = state.get("prev_flat_grad")
        prev_loss = state.get("prev_loss")
        d = state.get("d")
        t = state.get("t")

        # Initialize CUDA buffers if needed
        v_dim = self._numel()
        batch_size = 1  # Single optimization problem

        if "cuda_buffers_initialized" not in state:
            device = self._params[0].device
            dtype = self._params[0].dtype

            state["rho_buffer"] = torch.zeros(
                history_size, batch_size, device=device, dtype=dtype
            )
            state["y_buffer"] = torch.zeros(
                history_size, batch_size, v_dim, device=device, dtype=dtype
            )
            state["s_buffer"] = torch.zeros(
                history_size, batch_size, v_dim, device=device, dtype=dtype
            )
            state["step_vec"] = torch.zeros(
                batch_size, v_dim, device=device, dtype=dtype
            )
            state["q"] = torch.zeros(batch_size, v_dim, device=device, dtype=dtype)
            state["cuda_buffers_initialized"] = True

        rho_buffer = state["rho_buffer"]
        y_buffer = state["y_buffer"]
        s_buffer = state["s_buffer"]
        step_vec = state["step_vec"]
        q = state["q"]

        n_iter = 0

        while n_iter < max_iter:
            n_iter += 1
            state["n_iter"] += 1

            # Compute gradient descent direction using CUDA L-BFGS
            if state["n_iter"] == 1:
                # First iteration: use steepest descent
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1.0
            else:
                # Update history from previous step
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)  # Previous step direction * step size

                ys = y.dot(s)

                if ys > 1e-10:
                    # Update memory
                    if len(old_dirs) == history_size:
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)
                    H_diag = ys / y.dot(y)

                # Update CUDA buffers with current history
                num_old = len(old_dirs)
                if num_old > 0:
                    # Fill buffers: old_dirs[0] (oldest) goes to index 0, old_dirs[-1] (newest) goes to highest index
                    # The CUDA kernel processes them in order
                    m_actual = min(num_old, history_size)
                    for i in range(m_actual):
                        # Store oldest first, newest last
                        src_idx = num_old - m_actual + i
                        y_buffer[i, 0] = old_dirs[src_idx]
                        s_buffer[i, 0] = old_stps[src_idx]
                        rho_buffer[i, 0] = ro[src_idx]

                # Prepare inputs for CUDA kernel
                grad_q = flat_grad.unsqueeze(0)  # [1, v_dim]

                # Use previous values
                if prev_flat_grad is not None:
                    grad_0 = prev_flat_grad.unsqueeze(0)
                else:
                    grad_0 = grad_q.clone()

                if "prev_flat_param" in state:
                    x_0 = state["prev_flat_param"].unsqueeze(0)
                else:
                    x_0 = self._gather_flat_params().unsqueeze(0)

                # Compute step direction using CUDA kernel
                try:
                    step_direction = lbfgs_cuda(
                        step_vec=step_vec,
                        rho_buffer=rho_buffer,
                        y_buffer=y_buffer,
                        s_buffer=s_buffer,
                        q=q,
                        grad_q=grad_q,
                        x_0=x_0,
                        grad_0=grad_0,
                        epsilon=epsilon,
                        batch_size=batch_size,
                        m=min(num_old, history_size),
                        v_dim=v_dim,
                        stable_mode=stable_mode,
                        use_shared_buffers=use_shared_buffers,
                    )
                    d = step_direction[0]  # Remove batch dimension
                except Exception as e:
                    # Fallback to steepest descent if CUDA fails
                    print(f"Warning: CUDA L-BFGS failed, using steepest descent: {e}")
                    d = flat_grad.neg()

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone()
            else:
                prev_flat_grad.copy_(flat_grad)

            prev_loss = loss

            # Compute step length
            if state["n_iter"] == 1:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr

            gtd = flat_grad.dot(d)

            if gtd > -tolerance_change:
                break

            # Store current parameters before step
            prev_flat_param = self._gather_flat_params()

            # Apply step
            self._add_grad(t, d)

            # Re-evaluate function
            if n_iter != max_iter:
                loss = closure().item()
                flat_grad = self._gather_flat_grad()
                opt_cond = flat_grad.abs().max() <= tolerance_grad
                current_evals += 1
                state["func_evals"] += 1

            # Check termination conditions
            if n_iter == max_iter:
                break
            if current_evals >= max_eval:
                break
            if opt_cond:
                break
            if d.mul(t).abs().max() <= tolerance_change:
                break
            if abs(loss - prev_loss) < tolerance_change:
                break

        # Save state
        state["old_dirs"] = old_dirs
        state["old_stps"] = old_stps
        state["ro"] = ro
        state["H_diag"] = H_diag
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = prev_loss
        state["prev_flat_param"] = self._gather_flat_params()
        state["d"] = d
        state["t"] = t

        return orig_loss


# Alias for convenience
LBFGS_CUDA = LBFGSCuda


if __name__ == "__main__":
    # Simple test case
    model = torch.nn.Linear(10, 1).cuda()
    optimizer = LBFGSCuda(model.parameters(), lr=1.0, history_size=10)
