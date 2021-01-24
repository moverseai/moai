import torch
import functools
import typing

def expand_dims(
    src:            torch.Tensor,
    dst:            torch.Tensor,
    start_index:    int=1,
) -> torch.Tensor:
    r"""
        Expands the source tensor to match the spatial dimensions of the destination tensor.
        
        Arguments:
            src (torch.Tensor): A tensor of [B, K, X(Y)(Z)] dimensions
            dst (torch.Tensor): A tensor of [B, X(Y)(Z), (D), (H), W] dimensions
            start_index (int, optional): An optional start index denoting the start of the spatial dimensions
        
        Returns:
            A torch.Tensor of [B, K, X(Y)(Z), (1), (1), 1] dimensions. 
    """
    return functools.reduce(
        lambda s, _: s.unsqueeze(-1), 
        [*dst.shape[start_index:]],
        src
    )

def dim_list(
    tensor:         torch.Tensor,
    start_index:    int=1,
) -> typing.List[int]:
    return list(range(start_index, len(tensor.shape)))

def dims(
    tensor:         torch.Tensor,
    start_index:    int=1
) -> torch.Tensor:
    return torch.Tensor([tensor.size()[start_index:]]).squeeze().to(tensor.device)