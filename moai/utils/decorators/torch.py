import functools

import torch
from packaging import version

# __all__ = ["register_decorators", "ensure_contiguous"]
__all__ = ["ensure_contiguous", "adaptive_custom_fwd"]


# def register_decorators(*decorators):

#     # @functools.wraps(func)
#     def register_wrapper(func):
#         for deco in decorators[::-1]:
#             func = deco(func)
#         func._decorators = decorators
#         return func

#     return register_wrapper


def adaptive_custom_fwd(device_type="cuda", cast_inputs=torch.float32):
    """
    A decorator that adapts torch.amp.custom_fwd based on PyTorch version.
    If the version is >= 2.4.0, it uses torch.amp.custom_fwd.
    Otherwise, it falls back to a no-op decorator.
    """

    def decorator(fn):
        if version.parse(torch.__version__) >= version.parse("2.4.0"):
            fwd_decorator = torch.amp.custom_fwd(
                device_type=device_type, cast_inputs=cast_inputs
            )
            return fwd_decorator(fn)

        # PyTorch >= 1.6 and < 2.1: try torch.cuda.amp.custom_fwd
        elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "custom_fwd"):
            return torch.cuda.amp.custom_fwd(cast_inputs=cast_inputs)(fn)
        else:

            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapped

    return decorator


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
