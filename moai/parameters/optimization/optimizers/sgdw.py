import torch

__all__ = ['SGDW']

#NOTE: modified from https://github.com/wbaek/torchskeleton/tree/master/skeleton/optim

class SGDW(torch.optim.SGD):
    """Implements SGD with weight decay

    - **Paper**: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    - **Implementation**: [GitHub @ wbaek](https://github.com/wbaek/torchskeleton/tree/master/skeleton/optim)
    
    """
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

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # decoupled weight decay
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                p.data.add_(-group['lr'], d_p)

        return loss
