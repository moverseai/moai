import math
import torch
import torch
import typing

__all__ = ['AdamAIO']

#NOTE: modified from https://github.com/kayuksel/pytorch-adamaio

class AdamAIO(torch.optim.Optimizer):
    """Implements the All-In-One Adam Optimizer.

        The All-In-One Adam Optimizer, where several novelties are combined from following papers:

        - [Decoupled Weight Decay Regularization for Adam](https://arxiv.org/abs/1711.05101)

            Authors shown that the real reason why Momentum optimizer is often outperforming Adam in generalization was due to 
            the fact that Adam does not perform well under L2 regularization and developed decoupled weight decay as a solution.

        - [Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782)

            This is enabled via "hypergrad" parameter by setting it to any value except zero. It enables the optimizer to update
            the learning-rate itself by the technique proposed in the paper, instead of giving an external schedule which would 
            require lots of additional hyperparameters. It is especially useful when one doesn't want to hypertune a schedule.

        - [Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks](https://arxiv.org/abs/1711.05101)

            This can be set by the "partial" parameter, which controls how likely the optimizer acts similar to Adam (1.0) and 
            SGD (0.0), which is very useful if hypertuned. One can also update (decay) this parameter online to switch between 
            Adam and SGD optimizers in an easy way, which has been recommended by previous research for a better generalization.
    """
    def __init__(self, 
        params: typing.Iterator[torch.nn.Parameter],
        lr: float=2e-3,
        betas: typing.Tuple[float, float]=(0.9, 0.999),
        eps: float=5e-3,
        weight_decay: float=5e-6,
        hypergrad: float=1e-7,
        partial: float=0.75,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hypergrad=hypergrad, partial=partial)
        super().__init__(params, defaults)
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['hypergrad'] > 0 and state['step'] > 1:
                    prev_bias_correction1 = 1 - beta1 ** (state['step'] - 1)
                    prev_bias_correction2 = 1 - beta2 ** (state['step'] - 1)
                    h = torch.dot(grad.view(-1), torch.div(exp_avg, exp_avg_sq.sqrt().add_(group['eps'])).view(-1)) * math.sqrt(prev_bias_correction2) / prev_bias_correction1
                    group['lr'] += group['hypergrad'] * h
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.addcdiv_(-step_size, exp_avg, denom**group['partial'])
                    p.data.sub_(decayed_weights)
                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom**group['partial'])
        return loss
