from lark import Tree, Transformer, Token, v_args

import numpy as np
import torch
import dataclasses
import toolz
import typing
import benedict

@dataclasses.dataclass(unsafe_hash=True) #NOTE: needed for hashing https://stackoverflow.com/questions/60289768/error-unhashable-type-dict-with-dataclass
class NamedTensor(torch.nn.Module):
    key: str
    extracted: int
    index: int = -1

    def __post_init__(self):
        super().__init__()
    
    def forward(self, td, tmp) -> None:
        # keys = self.key.split('.') #TODO: update w/ benedict
        # value = toolz.get_in(self.key.split('.'), td)
        # tmp = toolz.assoc_in(tmp, keys, value)
        tmp[self.key] = td[self.key]
        # tmp[f'result{self.index}'] = value
        # return td, tmp
    
@dataclasses.dataclass(repr=False, unsafe_hash=True)
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
    
    def forward(self, td, tmp) -> None:
        lhs = toolz.get_in(self.lhs.split('.'), tmp)
        rhs = toolz.get_in(self.rhs.split('.'), tmp)
        if self.lhs_generated:
            lhs = lhs.to(rhs)
        if self.rhs_generated:
            rhs = rhs.to(lhs)
        tmp[f'result{self.index}'] = self.op(lhs, rhs)

@dataclasses.dataclass(repr=False, unsafe_hash=True)
class UnaryOperationTensors(torch.nn.Module):
    operation: str
    key: str
    index: int
    
    def __post_init__(self):
        super().__init__()
        self.op = getattr(torch, self.operation)
    
    def __repr__(self):
        return f"{self.operation}:{self.key}"
    
    def forward(self, td, tmp) -> None:
        tmp[f'result{self.index}'] = self.op(toolz.get_in(self.key.split('.'), tmp))
    
@dataclasses.dataclass(repr=False, unsafe_hash=True)
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
    
    def forward(self, td, tmp) -> None:
        tmp[f'result{self.index}'] = self.op(toolz.get_in(self.lhs.split('.'), tmp), self.rhs)

@dataclasses.dataclass(repr=False, unsafe_hash=True)
class NnaryOperationTensors(torch.nn.Module):
    operation: str
    keys: typing.List[str] = dataclasses.field(compare=False)
    dim: int
    index: int
    
    def __post_init__(self):
        super().__init__()
        self.op = getattr(torch, self.operation)
    
    def __repr__(self):
        return f"{self.operation}:{','.join(self.keys)}@{self.dim}"
    
    def forward(self, td, tmp) -> None: # torch.Tensor:
        tmp[f'result{self.index}'] = self.op([toolz.get_in(k.split('.'), tmp) for k in self.keys], dim=self.dim)
        # return td, tmp    

@dataclasses.dataclass(repr=False, unsafe_hash=True)
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
    
    def forward(self, td, tmp) -> None: # torch.Tensor:
        key = toolz.get_in(self.key.split('.'), tmp)
        if self.args is None:
            tmp[f'result{self.index}'] = self.op(key)
        else:
            tmp[f'result{self.index}'] = self.op(key, *self.args)
        # return td, tmp
        
@dataclasses.dataclass(repr=False, unsafe_hash=True)
class GenerationOperationTensors(torch.nn.Module):
    operation: str
    args: typing.Union[int, typing.Sequence[int]]
    index: int
    
    def __post_init__(self):
        super().__init__()
        self.op = getattr(torch, self.operation)
    
    def __repr__(self):
        return f"{self.operation}:{self.index}"
    
    def forward(self, td, tmp) -> None:
        tmp[f'result{self.index}'] = self.op(self.args)
    
@dataclasses.dataclass(repr=False, unsafe_hash=True)
class SlicingOperationTensors(torch.nn.Module):    
    operation: str
    key: str
    dim: int
    args: typing.Union[int, typing.Sequence[int]]
    index: int
    
    def __post_init__(self):
        super().__init__()
        self.op = getattr(torch, self.operation)
    
    def __repr__(self):
        return f"{self.operation}:{self.index}"
    
    def forward(self, td, tmp) -> None:
        t = toolz.get_in(self.key.split('.'), tmp)
        if isinstance(self.args, int):
            tmp[f'result{self.index}'] = self.op(t, self.dim, self.args)
        else:
            tmp[f'result{self.index}'] = self.op(t, self.dim, *self.args)

@dataclasses.dataclass(repr=False, unsafe_hash=True)
class IndexingOperationTensors(torch.nn.Module):    
    operation: str
    key: str
    dim: int
    indices: torch.Tensor
    index: int
    
    def __post_init__(self):
        super().__init__()
        self.op = getattr(torch, self.operation)
        self.register_buffer('idx', self.indices)
    
    def __repr__(self):
        return f"{self.operation}:{self.index}"
    
    def forward(self, td, tmp) -> None:
        t = toolz.get_in(self.key.split('.'), tmp)
        tmp[f'result{self.index}'] = self.op(t, self.dim, self.idx)

def unary(func):        
    def unary_wrapper(self, key):
        return self._unary(func.__name__, key)
    return unary_wrapper

def binary(func):        
    def binary_wrapper(self, lhs, rhs):
        return self._binary(func.__name__, lhs, rhs)
    return binary_wrapper

def binary_generated(func):        
    def binary_generated_wrapper(self, lhs, rhs):
        #NOTE: assumes the (torch) func name is suffixed only with a 'g'
        return self._binary_generated(func.__name__[:-1], lhs, rhs)
    return binary_generated_wrapper

def generation(func):        
    def generation_wrapper(self, *dims):
        return self._generation(func.__name__, *dims)
    return generation_wrapper

def nnary(func):        
    def nnary_wrapper(self, keys, dim):
        return self._nnary(func.__name__, keys, dim)
    return nnary_wrapper

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
        tmp = benedict.benedict({})
        for m in self.seq:            
            m(tensors, tmp) # tensors, tmp = m(tensors, tmp)
        tensors[self.key] = tmp[f'result{m.index}' if m.index >= 0 else m.key]        
        return tensors #NOTE: what if only extracted?

    def number(self, value):
        return float(value.value) # torch.scalar_tensor(float(value.value), dtype=torch.float32)

    def _binary(self, name, lhs, rhs):        
        if lhs is None:
            lhs = self.results.pop()
        if rhs is None:            
            rhs = self.results.pop()
        if not isinstance(lhs, str):
            m = BinaryOperationScalar(name, rhs, lhs, self.index)
        elif not isinstance(rhs, str):
            m = BinaryOperationScalar(name, lhs, rhs, self.index)
        else:
            m = BinaryOperationTensors(name, lhs, rhs, self.index)
        self.seq.add_module(f'{name}{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    @binary
    def add(self, lhs, rhs):
        pass

    @binary
    def sub(self, lhs, rhs):
        pass

    @binary
    def mul(self, lhs, rhs):
        pass

    @binary
    def div(self, lhs, rhs):
        pass

    @binary
    def pow(self, lhs, rhs):
        pass

    # def add(self, lhs, rhs):
    #     if lhs is None:
    #         lhs = self.results.pop()
    #     if rhs is None:            
    #         rhs = self.results.pop()
    #     if not isinstance(lhs, str):
    #         m = BinaryOperationScalar('add', rhs, lhs, self.index)
    #     elif not isinstance(rhs, str):
    #         m = BinaryOperationScalar('add', lhs, rhs, self.index)
    #     else:
    #         m = BinaryOperationTensors('add', lhs, rhs, self.index)
    #     self.seq.add_module(f'add{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    # def sub(self, lhs, rhs):
    #     if lhs is None: #NOTE: prev?
    #         lhs = self.results.pop()
    #     if rhs is None:
    #         rhs = self.results.pop()
    #     if not isinstance(lhs, str):
    #         m = BinaryOperationScalar('sub', rhs, lhs, self.index)
    #     elif not isinstance(rhs, str):
    #         m = BinaryOperationScalar('sub', lhs, rhs, self.index)
    #     else:
    #         m = BinaryOperationTensors('sub', lhs, rhs, self.index)
    #     self.seq.add_module(f'sub{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    # def mul(self, lhs, rhs):
    #     if lhs is None:
    #         lhs = self.results.pop()
    #     if rhs is None:
    #         rhs = self.results.pop()
    #     if not isinstance(lhs, str):
    #         m = BinaryOperationScalar('mul', rhs, lhs, self.index)
    #     elif not isinstance(rhs, str):
    #         m = BinaryOperationScalar('mul', lhs, rhs, self.index)
    #     else:
    #         m = BinaryOperationTensors('mul', lhs, rhs, self.index)
    #     self.seq.add_module(f'mul{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    # def div(self, lhs, rhs):
    #     if lhs is None:
    #         lhs = self.results.pop()
    #     if rhs is None:
    #         rhs = self.results.pop()
    #     if not isinstance(lhs, str):
    #         m = BinaryOperationScalar('div', rhs, lhs, self.index)
    #     elif not isinstance(rhs, str):
    #         m = BinaryOperationScalar('div', lhs, rhs, self.index)
    #     else:
    #         m = BinaryOperationTensors('div', lhs, rhs, self.index)
    #     self.seq.add_module(f'div{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    # def pow(self, lhs, rhs): #NOTE: check float_power()
    #     if lhs is None:
    #         lhs = self.results.pop()
    #     if rhs is None:
    #         rhs = self.results.pop()
    #     if not isinstance(lhs, str):
    #         m = BinaryOperationScalar('pow', rhs, lhs, self.index)
    #     elif not isinstance(rhs, str):
    #         m = BinaryOperationScalar('pow', lhs, rhs, self.index)
    #     else:
    #         m = BinaryOperationTensors('pow', lhs, rhs, self.index)
    #     self.seq.add_module(f'pow{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    def _binary_generated(self, name, lhs, rhs):        
        if lhs is None: #NOTE: only one of lhs/rhs should be generated
            lhs = self.results.pop()
            m = BinaryOperationTensors(name, lhs, rhs, self.index, True, False)
        if rhs is None: #NOTE: only one of lhs/rhs should be generated
            rhs = self.results.pop()
            m = BinaryOperationTensors(name, lhs, rhs, self.index, False, True)
        #NOTE: lhs being a scalar is an error, cant derive device        
        self.seq.add_module(f'{name}{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    @binary_generated
    def addg(self, lhs, rhs):
        pass

    @binary_generated
    def subg(self, lhs, rhs):
        pass
    
    @binary_generated
    def mulg(self, lhs, rhs):
        pass
    
    @binary_generated
    def divg(self, lhs, rhs):
        pass
    
    # def addg(self, lhs, rhs):
    #     if lhs is None: #NOTE: only one of lhs/rhs should be generated
    #         lhs = self.results.pop()
    #         m = BinaryOperationTensors('add', lhs, rhs, self.index, True, False)
    #     if rhs is None: #NOTE: only one of lhs/rhs should be generated
    #         rhs = self.results.pop()
    #         m = BinaryOperationTensors('add', lhs, rhs, self.index, False, True)
    #     #NOTE: lhs being a scalar is an error, cant derive device
    #     # if not isinstance(lhs, str):
    #     #     m = BinaryOperationScalar('mul', rhs, lhs, self.index)
    #     # else:
    #     self.seq.add_module(f'add{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    # def subg(self, lhs, rhs):
    #     if lhs is None: #NOTE: only one of lhs/rhs should be generated
    #         lhs = self.results.pop()
    #         m = BinaryOperationTensors('sub', lhs, rhs, self.index, True, False)
    #     if rhs is None: #NOTE: only one of lhs/rhs should be generated
    #         rhs = self.results.pop()
    #         m = BinaryOperationTensors('sub', lhs, rhs, self.index, False, True)
    #     #NOTE: lhs being a scalar is an error, cant derive device
    #     # if not isinstance(lhs, str):
    #     #     m = BinaryOperationScalar('mul', rhs, lhs, self.index)
    #     # else:
    #     self.seq.add_module(f'sub{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    # def mulg(self, lhs, rhs):
    #     if lhs is None: #NOTE: only one of lhs/rhs should be generated
    #         lhs = self.results.pop()
    #         m = BinaryOperationTensors('mul', lhs, rhs, self.index, True, False)
    #     if rhs is None: #NOTE: only one of lhs/rhs should be generated
    #         rhs = self.results.pop()
    #         m = BinaryOperationTensors('mul', lhs, rhs, self.index, False, True)
    #     #NOTE: lhs being a scalar is an error, cant derive device
    #     # if not isinstance(lhs, str):
    #     #     m = BinaryOperationScalar('mul', rhs, lhs, self.index)
    #     # else:
    #     self.seq.add_module(f'mul{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    # def divg(self, lhs, rhs):
    #     if lhs is None: #NOTE: only one of lhs/rhs should be generated
    #         lhs = self.results.pop()
    #         m = BinaryOperationTensors('div', lhs, rhs, self.index, True, False)
    #     if rhs is None: #NOTE: only one of lhs/rhs should be generated
    #         rhs = self.results.pop()
    #         m = BinaryOperationTensors('div', lhs, rhs, self.index, False, True)
    #     #NOTE: lhs being a scalar is an error, cant derive device
    #     # if not isinstance(lhs, str):
    #     #     m = BinaryOperationScalar('mul', rhs, lhs, self.index)
    #     # else:
    #     self.seq.add_module(f'div{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    def _generation(self, name, *dims):
        dims = list(map(int, dims))
        m = GenerationOperationTensors(name, dims, self.index)
        self.seq.add_module(f'{name}{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    @generation
    def zeros(self, *dims):
        pass

    @generation
    def ones(self, *dims):
        pass

    @generation
    def rand(self, *dims):
        pass

    @generation
    def randn(self, *dims):
        pass

    # def zeros(self, *dims):
    #     dims = list(map(int, dims))
    #     m = GenerationOperationTensors('zeros', dims, self.index)
    #     self.seq.add_module(f'zeros{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    # def ones(self, *dims):
    #     dims = list(map(int, dims))
    #     m = GenerationOperationTensors('ones', dims, self.index)
    #     self.seq.add_module(f'ones{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    # def rand(self, *dims):
    #     dims = list(map(int, dims))
    #     m = GenerationOperationTensors('rand', dims, self.index)
    #     self.seq.add_module(f'rand{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    # def randn(self, *dims):
    #     dims = list(map(int, dims))
    #     m = GenerationOperationTensors('randn', dims, self.index)
    #     self.seq.add_module(f'randn{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1    

    def _unary(self, name, key):        
        if key is None:
            key = self.results.pop()
        else:
            key = self.extract(key)
        m = UnaryOperationTensors(name, key, self.index)
        self.seq.add_module(f'{name}{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    @unary
    def neg(self, key):
        pass

    @unary
    def exp(self, key):
        pass

    @unary
    def log(self, key):
        pass

    @unary
    def reciprocal(self, key):
        pass

    @unary
    def sin(self, key):
        pass

    @unary
    def asin(self, key):
        pass

    @unary
    def cos(self, key):
        pass

    @unary
    def acos(self, key):
        pass

    @unary
    def tan(self, key):
        pass

    @unary
    def atan(self, key):
        pass

    @unary
    def abs(self, key):
        pass

    @unary
    def abs(self, key):
        pass

    @unary
    def zeros_like(self, key):
        pass

    @unary
    def ones_like(self, key):
        pass

    @unary
    def rand_like(self, key):
        pass

    @unary
    def randn_like(self, key):
        pass
    
    @unary
    def rad2deg(self, key):
        pass

    @unary
    def deg2rad(self, key):
        pass

    # def assign_var(self, name, value):
    #     self.td[name] = torch.scalar_tensor(value, dtype=torch.float32)
    #     return value

    def _extract_key(self, token_or_rule) -> str:
        if isinstance(token_or_rule, Token): #NOTE: is FIELD
            key = str(token_or_rule)
        elif isinstance(token_or_rule, str):
            key = token_or_rule
        else: #NOTE: is RULE
            # key = toolz.reduce(lambda l,r:f"{'' if not l else l.value}{'' if not r else '.' + r.value}", name.children)
            key = ".".join(map(str, filter(lambda x: x, token_or_rule.children)))
        return key
    
    def extract(self, name):
        key = self._extract_key(name)    
        self.seq.add_module(f"extract{self.extracted}", NamedTensor(key, self.extracted))
        self.extracted += 1
        return key
        
    def list(self, array):
        return sum(array.children)
    
    def names(self, *args):
        keys = list(map(self._extract_key, args))
        already_extracted = set(map(lambda m: m.key, filter(lambda m: isinstance(m, NamedTensor), self.seq)))
        to_extract = set(keys) - already_extracted
        for k in to_extract:
            self.extract(k)
        return list(keys)
    
    def _nnary(self, name, keys, dim):
        m = NnaryOperationTensors(name, keys, int(str(dim)), self.index)
        self.seq.add_module(f'{name}{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    @nnary
    def cat(self, keys, dim):
        pass

    @nnary
    def stack(self, keys, dim):
        pass
    
    # def cat(self, keys, dim):
    #     m = NnaryOperationTensors('cat', keys, int(str(dim)), self.index)
    #     self.seq.add_module(f'cat{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    # def stack(self, keys, dim):
    #     m = NnaryOperationTensors('stack', keys, int(str(dim)), self.index)
    #     self.seq.add_module(f'stack{self.index}', m)
    #     self.results.append(f'result{self.index}')
    #     self.index += 1

    def _transform_operation(self, name, key, dims):
        m = TransformOperationTensors(name, key, dims, self.index)
        self.seq.add_module(f'{name}{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def reshape(self, key, *dims):
        # if not isinstance(key, str): #NOTE: is lark.Tree
        key = self.extract(key)
        dims = list(map(int, dims))
        self._transform_operation('reshape', key, [dims])
        # m = TransformOperationTensors('reshape', key, [dims], self.index)
        # self.seq.add_module(f'reshape{self.index}', m)
        # self.results.append(f'result{self.index}')
        # self.index += 1

    def transpose(self, key, *dims):
        # if not isinstance(key, str): #NOTE: is lark.Tree
        key = self.extract(key)
        dims = list(map(int, dims))
        op = 'transpose' if len(dims) == 2 else 'permute'
        if len(dims) != 2: #NOTE: permute (similar to reshape) needs a list
            dims = [dims] 
        self._transform_operation(op, key, dims)
        # m = TransformOperationTensors(op, key, dims, self.index)
        # self.seq.add_module(f'{op}{self.index}', m)
        # self.results.append(f'result{self.index}')
        # self.index += 1

    def flatten(self, key, *dims):
        # if not isinstance(key, str): #NOTE: is lark.Tree
        key = self.extract(key)
        if dims[-1] is None:
            dims = [dims[0], -1]
        dims = list(map(int, dims))
        self._transform_operation('flatten', key, dims)
        # m = TransformOperationTensors('flatten', key, dims, self.index)
        # self.seq.add_module(f'flatten{self.index}', m)
        # self.results.append(f'result{self.index}')
        # self.index += 1

    def unsqueeze(self, key, *dims):
        if not isinstance(key, str): #NOTE: is lark.Tree
            key = self.extract(key)
        dims = list(map(int, dims))
        # if len(dims) == 1:
        #     dims = dims[0]
        self._transform_operation('unsqueeze', key, dims)
        # m = TransformOperationTensors('unsqueeze', key, dims, self.index)
        # self.seq.add_module(f'unsqueeze{self.index}', m)
        # self.results.append(f'result{self.index}')
        # self.index += 1

    def squeeze(self, key, *dims):
        if not isinstance(key, str): #NOTE: is lark.Tree
            key = self.extract(key)        
        dims = list(map(int, dims))
        # if len(dims) == 1:
        #     dims = dims[0]
        # elif len(dims) == 0:
        if len(dims) == 0:
            dims = None
        self._transform_operation('squeeze', key, dims)
        # m = TransformOperationTensors('squeeze', key, dims, self.index)
        # self.seq.add_module(f'squeeze{self.index}', m)
        # self.results.append(f'result{self.index}')
        # self.index += 1

    def _index(self, key, dim, index):
        if key is None:
            key = self.results.pop()
        m = SlicingOperationTensors('select', key, dim, index, self.index)
        self.seq.add_module(f'select{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def _indices(self, key, dim, indices):
        if key is None:
            key = self.results.pop()
        m = IndexingOperationTensors('index_select', key, dim, indices, self.index)
        self.seq.add_module(f'select{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def _slice(self, key, dim, range):
        if key is None:
            key = self.results.pop()
        start = range[0]
        length = range[1] - range[0]
        if start < 0:
            length *= -1        
        m = SlicingOperationTensors('narrow', key, dim, (start, length), self.index)
        self.seq.add_module(f'narrow{self.index}', m)
        self.results.append(f'result{self.index}')
        self.index += 1

    def slicing(self, token_or_tree, *args):
        key = self.extract(token_or_tree)
        n_args = len(args)        
        if args[0] == '...': #NOTE: error if ELLIPSIS in other position (not 0)
            dims = list(reversed(range(-1, -n_args, -1)))
            args = args[1:]
        else:
            dims = list(reversed(range(n_args)))
            args = tuple(reversed(args))
        for i, (a, d) in enumerate(filter(lambda ad: ad[0] != ':', zip(args, dims))):
            k = None if i else key
            if isinstance(a, Token):
                self._index(k, d, int(a))
            elif a.data == 'indices':
                self._indices(k, d, torch.tensor(list(map(int, a.children))))
            elif a.data == 'slice':
                self._slice(k, d, list(map(int, a.children))) #NOTE: error if >2 children
            else:
                raise RuntimeError(f"Unexpected RULE {a.data} when parsing a `slicing` expression.")