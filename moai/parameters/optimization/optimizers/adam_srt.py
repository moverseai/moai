import functools
import math
import typing

import torch

__all__ = ["AdamSRT", "AdamS"]

# NOTE: from https://github.com/ymontmarin/adamsrt/blob/master/adamsrt/optimizers/adam_srt.py


class AdamSRT(torch.optim.Optimizer):
    """Implements the AdamSRT algorithm.

    - **Paper**: [Spherical Perspective on Learning with Batch Norm. New methods : AdamSRT](https://arxiv.org/pdf/2006.13382.pdf)
    - **Implementation**: [GitHub @ ymontmarin](https://github.com/ymontmarin/adamsrt)

    General version of Adam-SRT that works for different normalization layer
    if specific channel options (channel_dims, channel_wise, channel_gloabal)
    are given.
    It should be used on parameters that are subject to scale invariance
    because they are followed by a normalization layer.
    Because not all params are concern, group_parameters of pytorch
    should be used.
    The effect is to adapt moments of Adam to the geometry implied by
    normalization layer. RT transform the order one moment ; make the
    order 2 moment rescaled and by norm.

    Example:
        >>> par_groups = [{'params': model.conv_params(), 'channel_wise'=True},
        >>>               {'params': model.other_params()}]
        >>> optimizer = AdamSRT(par_groups, lr=0.01, betas=(0.9, 0.9999))
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Arguments:
        params (list of dict or iterator): either a list of group_parameters
            which is a dict with key 'params' with a param iterator and other
            key for optimizer parameters overiding for this group of param,
        lr (float): learning rate,
        betas (tuple(float)): momentum factor for Adam moments,
        eps (float): float to avoid numerical instality in normalization,
        weight_decay (float): value for L2 regularization,
        channel_dims (list of int): the index of shape that represent the
            distinct dim that are independently normalized. Default value is
            channel_dims=shape which correspond to classic Adam.
            It can be used to adapt Adam to any normalization layers that
            follow conv layers,
        channel_wise (bool): if True and channel_dims is None set it to [0]
            which correspond to classic channel shape in 2D conv Network.
            Normalization will be done over other dims which are subject to
            scale invariance thanks to following normalization layer,
    """

    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: typing.Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        channel_dims=None,  # For customize the dimension for group of param
        channel_wise: bool = True,  # For default conv followed by BN invariance
        rt: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            channel_dims=channel_dims,
            channel_wise=channel_wise,
            rt=rt,
        )
        super(AdamSRT, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimizatino step
        """
        for group in self.param_groups:
            for p in group["params"]:

                # Get grad of params to update
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                if group["weight_decay"] != 0.0:
                    grad.add_(p.data, alpha=group["weight_decay"])

                # Get state
                state = self.state[p]

                # State initialization if needed
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    # Create the scalar product in respect of channel dims
                    shape = p.data.shape
                    channel_dims = group["channel_dims"]
                    if channel_dims is None:
                        if group["channel_wise"]:
                            # Classic meaning of channels
                            channel_dims = [0]
                        else:
                            # element wise : every element is a channel
                            # It correpond to classic adam update
                            channel_dims = list(range(len(shape)))
                    state["channel_dims"] = channel_dims
                    state["shape"] = shape

                # Start by increment step
                state["step"] += 1

                # Create the appropriate dot operator for the invar groups
                dot_ope = self.get_dot_operator(state["channel_dims"], state["shape"])

                # Retrive moments and constant
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                b1, b2 = group["betas"]

                # Update momentums
                exp_avg.mul_(b1).add_(grad, alpha=1 - b1)
                exp_avg_sq.mul_(b2).add_(dot_ope(grad, grad), alpha=1 - b2)
                # It should be d^-1 * (1 - beta2) instead
                # To avoid double div with > 1 we multiply stepsize by d^1/2

                # Compute bias correction
                bias_correction1 = 1 - b1 ** state["step"]
                bias_correction2 = 1 - b2 ** state["step"]

                # Compute actual nom and denom for the step
                nom = exp_avg / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )

                # Prepare data temporary copy for RT transform if needed
                if dot_ope.dim > 1 and group["rt"]:
                    prev_data = p.data.clone().detach()

                # Take the step on the datas
                step_size = group["lr"] * math.sqrt(dot_ope.dim)
                p.data.addcdiv_(nom, denom, value=-step_size)

                # We are on a sphere, we do RT transform
                if dot_ope.dim > 1 and group["rt"]:
                    new_data = p.data
                    prev_norm_sq = dot_ope(prev_data, prev_data)
                    new_norm_sq = dot_ope(new_data, new_data)
                    scal_x1_x2 = dot_ope(prev_data, new_data)
                    scal_m_x2 = dot_ope(exp_avg, new_data)
                    # R  order 2 moment
                    (exp_avg_sq.mul_(prev_norm_sq).div_(new_norm_sq + group["eps"]))
                    # RT the order 1 moment
                    (
                        exp_avg.mul_(scal_x1_x2)
                        .add_(-scal_m_x2 * prev_data)
                        .div_(new_norm_sq + group["eps"])
                    )

    @staticmethod
    def get_dot_operator(channel_dims, shape):
        """
        Generate a function that do scalar product for each channel dims
        Over the remaining dims
        """
        # Other dims are the ones of groups of elem for each channel
        grp_dims = list(set(range(len(shape))) - set(channel_dims))

        # Compute shape and size
        channel_shape = [shape[i] for i in channel_dims]
        grp_shape = [shape[i] for i in grp_dims]
        channel_size = functools.reduce(lambda x, y: x * y, [1] + channel_shape)
        grp_size = functools.reduce(lambda x, y: x * y, [1] + grp_shape)

        # Prepare the permutation to ordonate dims and its reciproc
        perm = channel_dims + grp_dims
        antiperm = [e[1] for e in sorted([(j, i) for i, j in enumerate(perm)])]

        # Prepare index query that retrieve all dimensions
        slice_len = max(len(channel_shape), 1)
        idx = [slice(None)] * slice_len + [None] * (len(shape) - slice_len)

        # Define the scalar product channel wise over grp dims
        # Output have is extend to fit initial shape
        def scalar_product(tensor1, tensor2):
            return (
                (tensor1 * tensor2)
                .permute(perm)  # permute as chan_dims, grp_dims
                .contiguous()
                .view(channel_size, grp_size)  # view as 2 dims tensor
                .sum(dim=1)  # norm over group dims to have scalar
                .view(*(channel_shape if channel_shape else [-1]))[
                    idx
                ]  # restore channel shape and extend on grp dims
                .permute(antiperm)  # Reverse permute to retrieve shape
                .contiguous()
            )

        scalar_product.dim = grp_size
        return scalar_product


class AdamS(AdamSRT):
    def __init__(
        self,
        params: typing.Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: typing.Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        channel_dims=None,  # For customize the dimension for group of param
        channel_wise: bool = True,  # For default conv followed by BN invariance
    ):
        super(AdamS, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            channel_dims=channel_dims,
            channel_wise=channel_wise,
            rt=False,  # Never do RT
        )
