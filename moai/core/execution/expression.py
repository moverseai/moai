from lark import Tree, Transformer, v_args

import numpy as np
import torch
import dataclasses
import toolz
import typing

@dataclasses.dataclass(unsafe_hash=True) #NOTE: needed for hashing https://stackoverflow.com/questions/60289768/error-unhashable-type-dict-with-dataclass
class NamedTensor(torch.nn.Module):
    key: str
    extracted: int
    index: int = -1

    def __post_init__(self):
        super().__init__()
    
    def forward(self, td, tmp) -> torch.Tensor:
        keys = self.key.split('.') #TODO: update w/ benedict
        value = toolz.get_in(self.key.split('.'), td)
        tmp = toolz.assoc_in(tmp, keys, value)
        # tmp[f'result{self.index}'] = value
        return td, tmp
    
@dataclasses.dataclass(repr=False)
class BinaryOperationTensors(torch.nn.Module):
    operation: str
    lhs: str
    rhs: str
    index: int
    lhs_generated: bool = False # dataclasses.field(default=False)
    rhs_generated: bool = False # dataclasses.field(default=False)
    
    def __post_init__(self):
        super().__init__()        
        self.op = getattr(torch, self.operation)
    
    def __repr__(self):
        return f"{self.operation}:{self.lhs},{self.rhs}"
    
    def forward(self, td, tmp) -> torch.Tensor:
        lhs = toolz.get_in(self.lhs.split('.'), tmp)
        rhs = toolz.get_in(self.rhs.split('.'), tmp)
        if self.lhs_generated:
            lhs = lhs.to(rhs)
        if self.rhs_generated:
            rhs = rhs.to(lhs)
        tmp[f'result{self.index}'] = self.op(lhs, rhs)
        return td, tmp

@dataclasses.dataclass(repr=False)
class UnaryOperationTensors(torch.nn.Module):
    operation: str
    key: str
    index: int
    
    def __post_init__(self):
        super().__init__()
        self.op = getattr(torch, self.operation)
    
    def __repr__(self):
        return f"{self.operation}:{self.key}"
    
    def forward(self, td, tmp) -> torch.Tensor:
        tmp[f'result{self.index}'] = self.op(toolz.get_in(self.key.split('.'), tmp))
        return td, tmp
    
@dataclasses.dataclass(repr=False)
class BinaryOperationScalar(torch.nn.Module):
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
class NnaryOperationTensors(torch.nn.Module):
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
    

@dataclasses.dataclass(repr=False)
class TransformOperationTensors(torch.nn.Module):
    operation: str
    key: str
    args: typing.Union[int, typing.Sequence[int]]
    index: int
    
    def __post_init__(self):
        super().__init__()
        self.op = getattr(torch, self.operation)
    
    def __repr__(self):
        return f"{self.operation}:{self.key}"
    
    def forward(self, td, tmp) -> torch.Tensor:        
        tmp[f'result{self.index}'] = self.op(toolz.get_in(self.key.split('.'), tmp), self.args)
        return td, tmp
        
@dataclasses.dataclass(repr=False)
class GenerationOperationTensors(torch.nn.Module):
    operation: str
    args: typing.Union[int, typing.Sequence[int]]
    index: int
    
    def __post_init__(self):
        super().__init__()
        self.op = getattr(torch, self.operation)
    
    def __repr__(self):
        return f"{self.operation}:{self.index}"
    
    def forward(self, td, tmp) -> torch.Tensor:        
        tmp[f'result{self.index}'] = self.op(self.args)
        return td, tmp
    
@v_args(inline=True) # Affects the signatures of the methods
class TreeModule(torch.nn.Module, Transformer):
    def __init__(self, key: str, tree: Tree):
        super().__init__()
        self.seq = torch.nn.ModuleList()
        self.index = 0
        self.extracted = 0
        self.operands = []
        self.key = key
        self.results = []
        self.transform(tree) #NOTE: should be last        
    
    def forward(self, tensors):
        tmp = {}
        for m in self.seq:
            tensors, tmp = m(tensors, tmp)
        tensors[self.key] = tmp[f'result{m.index}' if m.index >= 0 else m.key]
        #NOTE: what if only extracted?
        return tensors

    def number(self, value):
        # return torch.scalar_tensor(float(value.value), dtype=torch.float32)
        return float(value.value)

    def add(self, lhs, rhs):
        # prev = -1
        if lhs is None:
            # lhs = f'result{self.index + prev}'
            # prev -= 1
            lhs = self.results.pop()
        if rhs is None:
            # if prev == -2: #NOTE: lhs was None
                # prev -= 1
            # rhs = f'result{self.index + prev}'
            rhs = self.results.pop()
        if not isinstance(lhs, str):
            m = BinaryOperationScalar('add', rhs, lhs, self.index)
        elif not isinstance(rhs, str):
            m = BinaryOperationScalar('add', lhs, rhs, self.index)
        else:
            m = BinaryOperationTensors('add', lhs, rhs, self.index)
        self.seq.add_module(f'add{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def sub(self, lhs, rhs):
        # prev = -1
        if lhs is None: #NOTE: prev?
            # lhs = f'result{self.index + prev}'
            lhs = self.results.pop()
            # prev -= 1
        if rhs is None:
            # rhs = f'result{self.index + prev}'
            rhs = self.results.pop()
        if not isinstance(lhs, str):
            m = BinaryOperationScalar('sub', rhs, lhs, self.index)
        elif not isinstance(rhs, str):
            m = BinaryOperationScalar('sub', lhs, rhs, self.index)
        else:
            m = BinaryOperationTensors('sub', lhs, rhs, self.index)
        self.seq.add_module(f'sub{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def zeros(self, *dims):
        dims = list(map(int, dims))
        m = GenerationOperationTensors('zeros', dims, self.index)
        self.seq.add_module(f'zeros{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def ones(self, *dims):
        dims = list(map(int, dims))
        m = GenerationOperationTensors('ones', dims, self.index)
        self.seq.add_module(f'ones{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def rand(self, *dims):
        dims = list(map(int, dims))
        m = GenerationOperationTensors('rand', dims, self.index)
        self.seq.add_module(f'rand{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def randn(self, *dims):
        dims = list(map(int, dims))
        m = GenerationOperationTensors('randn', dims, self.index)
        self.seq.add_module(f'randn{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def mul(self, lhs, rhs):
        # prev = -1
        if lhs is None:
            # lhs = f'result{self.index + prev}'
            # prev -= 1
            lhs = self.results.pop()
        if rhs is None:
            # if prev == -2: #NOTE: lhs was None
                # prev -= 1
            # rhs = f'result{self.index + prev}'
            rhs = self.results.pop()
        if not isinstance(lhs, str):
            m = BinaryOperationScalar('mul', rhs, lhs, self.index)
        elif not isinstance(rhs, str):
            m = BinaryOperationScalar('mul', lhs, rhs, self.index)
        else:
            m = BinaryOperationTensors('mul', lhs, rhs, self.index)
        self.seq.add_module(f'mul{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def mulg(self, lhs, rhs):        
        if lhs is None: #NOTE: only one of lhs/rhs should be generated
            lhs = self.results.pop()
            m = BinaryOperationTensors('mul', lhs, rhs, self.index, True, False)
        if rhs is None: #NOTE: only one of lhs/rhs should be generated
            rhs = self.results.pop()
            m = BinaryOperationTensors('mul', lhs, rhs, self.index, False, True)
        #NOTE: lhs being a scalar is an error, cant derive device
        # if not isinstance(lhs, str):
        #     m = BinaryOperationScalar('mul', rhs, lhs, self.index)
        # else:        
        self.seq.add_module(f'mul{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def div(self, lhs, rhs):
        # prev = -1
        if lhs is None:
            # lhs = f'result{self.index + prev}'
            # prev -= 1
            lhs = self.results.pop()
        if rhs is None:
            # if prev == -2: #NOTE: lhs was None
                # prev -= 1
            # rhs = f'result{self.index + prev}'
            rhs = self.results.pop()
        if not isinstance(lhs, str):
            m = BinaryOperationScalar('div', rhs, lhs, self.index)
        elif not isinstance(rhs, str):
            m = BinaryOperationScalar('div', lhs, rhs, self.index)
        else:
            m = BinaryOperationTensors('div', lhs, rhs, self.index)
        self.seq.add_module(f'div{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def divg(self, lhs, rhs):
        if lhs is None: #NOTE: only one of lhs/rhs should be generated
            lhs = self.results.pop()
            m = BinaryOperationTensors('div', lhs, rhs, self.index, True, False)
        if rhs is None: #NOTE: only one of lhs/rhs should be generated
            rhs = self.results.pop()
            m = BinaryOperationTensors('div', lhs, rhs, self.index, False, True)
        #NOTE: lhs being a scalar is an error, cant derive device
        # if not isinstance(lhs, str):
        #     m = BinaryOperationScalar('mul', rhs, lhs, self.index)
        # else:        
        self.seq.add_module(f'div{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def pow(self, lhs, rhs):
        # prev = -1
        if lhs is None:
            # lhs = f'result{self.index + prev}'
            # prev -= 1
            lhs = self.results.pop()
        if rhs is None:
            # if prev == -2: #NOTE: lhs was None
                # prev -= 1
            # rhs = f'result{self.index + prev}'
            rhs = self.results.pop()
        if not isinstance(lhs, str):
            m = BinaryOperationScalar('pow', rhs, lhs, self.index)
        elif not isinstance(rhs, str):
            m = BinaryOperationScalar('pow', lhs, rhs, self.index)
        else:
            m = BinaryOperationTensors('pow', lhs, rhs, self.index)
        self.seq.add_module(f'pow{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def neg(self, key):
        # prev = -1
        if key is None:
            # lhs = f'result{self.index + prev}'
            # prev -= 1
            key = self.results.pop()
        else:
            key = self.extract(key)
        m = UnaryOperationTensors('neg', key, self.index)
        self.seq.add_module(f'neg{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def zeros_like(self, key):
        # prev = -1
        if key is None:
            # lhs = f'result{self.index + prev}'
            # prev -= 1
            key = self.results.pop()
        else:
            key = self.extract(key)
        m = UnaryOperationTensors('zeros_like', key, self.index)
        self.seq.add_module(f'zeros_like{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def ones_like(self, key):
        # prev = -1
        if key is None:
            # lhs = f'result{self.index + prev}'
            # prev -= 1
            key = self.results.pop()
        else:
            key = self.extract(key)
        m = UnaryOperationTensors('ones_like', key, self.index)
        self.seq.add_module(f'ones_like{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def rand_like(self, key):
        # prev = -1
        if key is None:
            # lhs = f'result{self.index + prev}'
            # prev -= 1
            key = self.results.pop()
        else:
            key = self.extract(key)
        m = UnaryOperationTensors('rand_like', key, self.index)
        self.seq.add_module(f'rand_like{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def randn_like(self, key):
        # prev = -1
        if key is None:
            # lhs = f'result{self.index + prev}'
            # prev -= 1
            key = self.results.pop()
        else:
            key = self.extract(key)
        m = UnaryOperationTensors('randn_like', key, self.index)
        self.seq.add_module(f'randn_like{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    # def assign_var(self, name, value):
    #     self.td[name] = torch.scalar_tensor(value, dtype=torch.float32)
    #     return value

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
        m = NnaryOperationTensors('cat', keys, int(str(dim)), self.index)
        self.seq.add_module(f'cat{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def stack(self, keys, dim):
        m = NnaryOperationTensors('stack', keys, int(str(dim)), self.index)
        self.seq.add_module(f'stack{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def reshape(self, key, *dims):
        if not isinstance(key, str): #NOTE: is lark.Tree
            key = self.extract(key)
        dims = list(map(int, dims))
        m = TransformOperationTensors('reshape', key, dims, self.index)
        self.seq.add_module(f'reshape{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1