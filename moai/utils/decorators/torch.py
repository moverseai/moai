import functools

import torch

# __all__ = ["register_decorators", "ensure_contiguous"]
__all__ = ["ensure_contiguous"]


# def register_decorators(*decorators):

#     # @functools.wraps(func)
#     def register_wrapper(func):
#         for deco in decorators[::-1]:
#             func = deco(func)
#         func._decorators = decorators
#         return func

#     return register_wrapper


def ensure_contiguous(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = tuple(  # Make sure all arguments are contiguous
            (  # typically this is `self` if used on an object method
                arg.contiguous()
                if torch.is_tensor(arg) and not arg.is_contiguous()
                else arg
            )
            for arg in args
        )
        kwargs = {  # Ensure all keyword arguments are contiguous if they are tensors
            k: (v.contiguous() if torch.is_tensor(v) and not v.is_contiguous() else v)
            for k, v in kwargs.items()
        }

        return func(*args, **kwargs)  # call function w/ modified arguments

    return wrapper
