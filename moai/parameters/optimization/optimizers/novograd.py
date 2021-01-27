import math
import torch
import typing

__all__ = ['NovoGrad']

#NOTE: modified from https://github.com/lonePatient/NovoGrad-pytorch

class NovoGrad(torch.optim.Optimizer):
    """Implements NovoGrad algorithm.

    - **Paper**: [Training Deep Networks with Stochastic Gradient Normalized by Layerwise Adaptive Second Moments](https://arxiv.org/pdf/1905.11286.pdf)
    - **Implementation**: [GitHub @ lonePatient](https://github.com/lonePatient/NovoGrad-pytorch)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0.98))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    Example:
        >>> model = ResNet()
        >>> optimizer = NovoGrad(model.parameters(), lr=1e-2, weight_decay=1e-5)
    """

    def __init__(self, 
        params: typing.Iterator[torch.nn.Parameter],
        lr: float=0.01,
        betas: typing.Tuple[float, float]=(0.95, 0.98),
        eps: float=1e-8,
        weight_decay: float=0,
        grad_averaging: bool=False
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,weight_decay=weight_decay,grad_averaging = grad_averaging)
        super().__init__(params, defaults)

    def step(self, closure: typing.Optional[typing.Callable[[], float]]=None) -> typing.Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NovoGrad does not support sparse gradients')
                state = self.state[p]
                g_2 = torch.sum(grad ** 2)
                if len(state) == 0:
                    state['step'] = 0
                    state['moments'] = grad.div(g_2.sqrt() +group['eps']) + \
                                       group['weight_decay'] * p.data
                    state['grads_ema'] = g_2
                moments = state['moments']
                grads_ema = state['grads_ema']
                beta1, beta2 = group['betas']
                state['step'] += 1
                grads_ema.mul_(beta2).add_(1 - beta2, g_2)

                denom = grads_ema.sqrt().add_(group['eps'])
                grad.div_(denom)
                # weight decay
                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    grad.add_(decayed_weights)

                # Momentum --> SAG
                if group['grad_averaging']:
                    grad.mul_(1.0 - beta1)

                moments.mul_(beta1).add_(grad) # velocity

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.add_(-step_size, moments)

        return loss

class AdamW(torch.optim.Optimizer):
    """Implements Adam algorithm (with proper weight decay).

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    Example:
        >>> model = alexnet()
        >>> optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    """

    def __init__(self, 
        params: typing.Iterator[torch.nn.Parameter], 
        lr: float=1e-3,
        betas: typing.Tuple[float, float]=(0.9, 0.999),
        eps: float=1e-8,
        weight_decay: float=0,
        amsgrad: bool=False
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,weight_decay=weight_decay, amsgrad=amsgrad)
        #super(AdamW, self).__init__(params, defaults)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss