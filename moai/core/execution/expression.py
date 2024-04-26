from lark import Transformer, v_args

import numpy as np
import torch
import dataclasses
import toolz
import typing

@dataclasses.dataclass()
class NamedTensor(torch.nn.Module):
    key: str
    extracted: int
    index: int = -1

    def __post_init__(self):
        super().__init__()
    
    def forward(self, td, tmp) -> torch.Tensor:
        keys = self.key.split('.')
        value = toolz.get_in(self.key.split('.'), td)
        tmp = toolz.assoc_in(tmp, keys, value)
        # tmp[f'result{self.index}'] = value
        return td, tmp
    
@dataclasses.dataclass(repr=False)
class OperationTensors(torch.nn.Module): #binary
    operation: str
    lhs: str
    rhs: str
    index: int
    
    def __post_init__(self):
        super().__init__()
        self.op = getattr(torch, self.operation)
    
    def __repr__(self):
        return f"{self.operation}:{self.lhs},{self.rhs}"
    
    def forward(self, td, tmp) -> torch.Tensor:        
        tmp[f'result{self.index}'] = self.op(toolz.get_in(self.lhs.split('.'), tmp), toolz.get_in(self.rhs.split('.'), tmp))
        return td, tmp

@dataclasses.dataclass(repr=False)
class OperationScalar(torch.nn.Module): #unary
    operation: str
    lhs: str
    rhs: typing.Union[float, int]
    index: int
    
    def __post_init__(self):
        super().__init__()
        self.op = getattr(torch, self.operation)
    
    def __repr__(self):
        return f"{self.operation}:{self.lhs},{self.rhs}"
    
    def forward(self, td, tmp) -> torch.Tensor:
        tmp[f'result{self.index}'] = self.op(toolz.get_in(self.lhs.split('.'), tmp), self.rhs)
        return td, tmp

@dataclasses.dataclass(repr=False)
class AggregateOperationTensors(torch.nn.Module): #n-ary
    operation: str
    keys: typing.List[str]
    dim: int
    index: int
    
    def __post_init__(self):
        super().__init__()
        self.op = getattr(torch, self.operation)
    
    def __repr__(self):
        return f"{self.operation}:{','.join(self.keys)}@{self.dim}"
    
    def forward(self, td, tmp) -> torch.Tensor:
        tmp[f'result{self.index}'] = self.op([toolz.get_in(k.split('.'), tmp) for k in self.keys], dim=self.dim)
        return td, tmp
    

@v_args(inline=True) # Affects the signatures of the methods
class CalculateTreeTorch(torch.nn.Module, Transformer):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.ModuleList()
        self.index = 0
        self.extracted = 0
        self.operands = []
        #TODO: need key to add to the tensors
    
    def forward(self, tensors):
        tmp = {}
        for m in self.seq:
            tensors, tmp = m(tensors, tmp)
        tensors['result'] = tmp[f'result{m.index}' if m.index >= 0 else m.key]
        #NOTE: what if only extracted?
        return tensors

    def number(self, value):
        return torch.scalar_tensor(float(value.value), dtype=torch.float32)

    def add(self, lhs, rhs):
        prev = -1
        if lhs is None:
            lhs = f'result{self.index + prev}'
            prev -= 1
        if rhs is None:
            rhs = f'result{self.index + prev}'
        if not isinstance(lhs, str):
            m = OperationScalar('add', rhs, lhs, self.index)
        elif not isinstance(rhs, str):
            m = OperationScalar('add', lhs, rhs, self.index)
        else:
            m = OperationTensors('add', lhs, rhs, self.index)
        self.seq.add_module(f'add{self.index}', m)
        self.index += 1

    def sub(self, lhs, rhs):
        if lhs is None: #NOTE: prev?
            lhs = f'result{self.index-1}'
        if rhs is None:
            rhs = f'result{self.index-1}'
        if not isinstance(lhs, str):
            m = OperationScalar('sub', rhs, lhs, self.index)
        elif not isinstance(rhs, str):
            m = OperationScalar('sub', lhs, rhs, self.index)
        else:
            m = OperationTensors('sub', lhs, rhs, self.index)
        self.seq.add_module(f'sub{self.index}', m)
        self.index += 1

    def mul(self, lhs, rhs):
        pass

    def div(self, lhs, rhs):
        pass

    def neg(self, lhs, rhs):
        pass

    def assign_var(self, name, value):
        self.td[name] = torch.scalar_tensor(value, dtype=torch.float32)
        return value

    def extract(self, name):
        key = toolz.reduce(lambda l,r:f"{'' if not l else l.value}{'' if not r else '.' + r.value}", name.children)        
        self.seq.add_module(f"extract{self.extracted}", NamedTensor(key, self.extracted))
        self.extracted += 1
        return key
        
    def list(self, array):
        return sum(array.children)
    
    def names(self, *args):
        keys = []
        to_add = []
        for arg in args:
            if arg.data.type == 'RULE' and arg.data.value == 'name':
                k = toolz.reduce(lambda l,r:f"{'' if not l else l.value}{'' if not r else '.' + r.value}", arg.children)
                add = False
                for m in self.seq:
                    if isinstance(m, NamedTensor):                        
                        if m.key != k:
                            add = True
                            break
                to_add.append(arg)
                keys.append(k)
        for a in to_add:
            self.extract(a)
        return keys
    
    def cat(self, keys, dim):
        m = AggregateOperationTensors('cat', keys, int(str(dim)), self.index)
        self.seq.add_module(f'cat{self.index}', m)
        self.index += 1

    def stack(self, keys, dim):
        m = AggregateOperationTensors('stack', keys, int(str(dim)), self.index)
        self.seq.add_module(f'stack{self.index}', m)
        self.index += 1