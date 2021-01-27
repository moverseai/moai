import torch
import functools
import typing

__all__ = ['HypergradSGD']

#NOTE: modified from https://github.com/wbaek/torchskeleton/tree/master/skeleton/optim
#NOTE: code from https://github.com/gbaydin/hypergradient-descent/blob/master/hypergrad/sgd_hd.py

class HypergradSGD(torch.optim.Optimizer):
    r""" Implements the Hypergradient Descent Algorithm.

    - **Paper**: [ONLINE LEARNING RATE ADAPTATION WITH HYPERGRADIENT DESCENT](https://openreview.net/pdf?id=BkrsAzWAb)
    - **Implementation**: [GitHub @ gbaydin](https://github.com/gbaydin/hypergradient-descent/)

    """
    def __init__(self, 
        params: typing.Iterator[torch.nn.Parameter],
        #lr: torch.optim.optimizer.required,
        lr=None,
        momentum: float=0,
        dampening: float=0,
        weight_decay: float=0,
        nesterov: bool=False,
        hypergrad_lr: float=0,
        hypergrad_momentum: float=0,
        min_lr: float=1e-6
    ):
        #if lr is not torch.optim.optimizer.required and lr < 0.0:
        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        hypergrad_lr=hypergrad_lr, hypergrad_momentum=hypergrad_momentum, min_lr=min_lr)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(HypergradSGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group['prev_grads'] = [torch.zeros_like(p, requires_grad=False) for p in group['params']]
            group['lr_per_layers'] = [group['lr'] for p in group['params']]

    def __setstate__(self, state):
        super(HypergradSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            hypergrad_lr = group['hypergrad_lr']
            hypergrad_momentum = group['hypergrad_momentum']
            min_lr = group['min_lr']

            lr_per_layers = group['lr_per_layers']

            for index, (p, d_p_prev) in enumerate(zip(group['params'], group['prev_grads'])):
                if p.grad is None:
                    continue

                d_p = p.grad.data
                h = torch.dot(d_p.view(-1), d_p_prev.view(-1))
                # group['prev_grads'][index] = d_p.clone().detach().requires_grad_(False)

                if hypergrad_momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer_h' not in param_state:
                        buf_h = param_state['momentum_buffer_h'] = torch.clone(h).detach()
                    else:
                        buf_h = param_state['momentum_buffer_h']
                        buf_h.mul_(hypergrad_momentum).add_(1, h)
                    h = buf_h

                lr = lr_per_layers[index]
                lr = max(min_lr, lr + (hypergrad_lr * h))
                lr_per_layers[index] = lr

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                group['prev_grads'][index] = d_p.clone().detach().requires_grad_(False)
                p.data.add_(-lr, d_p)

                # Apply decoupled weight decay
                if weight_decay != 0:
                    p.data.add_(-lr, weight_decay)

            group['lr'] = sum(group['lr_per_layers']) / len(group['lr_per_layers'])

        return loss
