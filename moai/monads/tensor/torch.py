import torch
import typing

class Stack(torch.nn.Module):
    def __init__(self,
        dim: int=0
    ):
        super(Stack, self).__init__()
        self.dim = dim

    def forward(self, tensors: typing.List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tensors, dim=self.dim)

class Concat(torch.nn.Module):
    def __init__(self,
        dim: int=1
    ):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, tensors: typing.List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=self.dim)

class Split(torch.nn.Module): #TODO: optimize by returning the tuple
    def __init__(self,
        dim: int=0, # at which dim to split
        split: int=0, # how many to split to, with 0 denoting an even split
    ):
        super(Split, self).__init__()
        self.dim = dim
        self.split = split

    def forward(self, 
        tensor: torch.Tensor,
        # index: torch.Tensor # scalar tensor denoting the split index
    ) -> typing.Dict[str, torch.Tensor]:
    # ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        ret = {}
        size = self.split if self.split else tensor.shape[self.dim] // 2
        chunks = torch.split(tensor, size, dim=self.dim) #[int(index)]
        for i in range(len(chunks)):
            ret['chunk'+str(i)] = chunks[i]
        return ret


class Slice(torch.nn.Module):
    def __init__(self,
        dim: int=0, # at which dim to slice
        start : int=0, #the index to start with
        length: int=1, #the index to end with
    ):
        super(Slice, self).__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.narrow(input = tensor,dim=self.dim,start = self.start, length = self.length)



#TODO: Is this really needed?
class SelectTensor(torch.nn.Module):
    def __init__(self):

        super(SelectTensor,self).__init__()
    
    def forward(self,
                tensors: typing.List[torch.Tensor],
                ref_tensor: torch.Tensor) -> torch.Tensor:

                for ten in tensors:
                    if ten.shape[-2:] == ref_tensor.shape[-2:]:
                        out_tensor = ten
                        break
                    else:
                       out_tensor = None
                
                return out_tensor

class Detach(torch.nn.Module):
    def __init__(self):
        super(Detach, self).__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach()

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.flatten = torch.nn.Flatten()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.flatten(tensor)

class ReshapeAs(torch.nn.Module):
    def __init__(self):
        super(ReshapeAs, self).__init__()

    def forward(self,
        tensor: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        return tensor.reshape_as(shape)

class Identity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

# Alias = functools.partial(Identity)
# Passthrough = functools.partial(Identity)
Alias = Identity
Passthrough = Identity