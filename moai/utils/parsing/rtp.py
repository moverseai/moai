import torch
import pysm
import re
import toolz
import functools
import itertools
import inspect
import typing
import logging
import omegaconf.omegaconf

log = logging.getLogger(__name__)

__all__ = [
    'get_result_fillers',
    'resolve_io_config',
    'split_as',
    'force_list',
]

def _tuple_add(
    tensor_dict: typing.Dict[str, torch.Tensor],
    tensor: torch.Tensor,
    key: str
) -> None:
    tensor_dict[key] = tensor

def _list_add(
    tensor_dict: typing.Dict[str, torch.Tensor],
    tensor_list: typing.List[torch.Tensor],
    key: str
) -> None:
    for i, tensor in enumerate(tensor_list):
        tensor_dict[key + "_" + str(i)] = tensor

def _tensor_add(
    tensor_dict: typing.Dict[str, torch.Tensor], 
    tensor: torch.Tensor,
    key: str
) -> None:
    tensor_dict[key] = tensor

def _no_add(
    tensor_dict: typing.Dict[str, torch.Tensor],
    tensor: torch.Tensor
) -> None:
    pass

__ACTION_MAP__ = {
    "Tuple": _tuple_add,
    "List": _list_add,
    "Tensor": _tensor_add,
    "NoOp": _no_add
}

class ReturnTypeParser(object):
    def __init__(self, keys):
        self.keys = keys
        self.counter = 0
        self.sm = self._create_state_machine()
        self.result = []


    def _create_state_machine(self) -> pysm.StateMachine:
        sm = pysm.StateMachine('sm')

        initial = pysm.State('Initial')
        move_to = pysm.State('MoveTo')
        advance = pysm.State('Advance')
        begin = pysm.State('Begin')
        end = pysm.State('End')
        tensor = pysm.State('Tensor')
        sequence = pysm.State('Sequence')

        sm.add_state(initial, initial=True)
        sm.add_state(move_to)
        sm.add_state(advance)
        sm.add_state(begin)
        sm.add_state(end)
        sm.add_state(end)
        sm.add_state(tensor)
        sm.add_state(sequence)

        sm.add_transition(initial, move_to,
            events=['parse'], input=['Tuple', 'List'],
            action=self._set_func)
        sm.add_transition(initial, tensor,
            events=['parse'], input=['Tensor'],
            action=self._single_out)
        #TODO: from list to tuple before end should generate an error !
        sm.add_transition(move_to, begin, 
            events=['parse'], input='[',
            action=self._start)
        sm.add_transition(begin, tensor,
            events=['parse'], input=['Tensor'],
            action=self._register)
        sm.add_transition(advance, tensor,
            events=['parse'], input=['Tensor'],
            action=self._register)
        sm.add_transition(tensor, advance,
            events=['parse'], input=',',
            action=self._advance)
        sm.add_transition(end, advance,
            events=['parse'], input=',',
            action=self._advance)
        sm.add_transition(advance, move_to,
            events=['parse'], input=['Tuple', 'List'],
            action=self._set_func)
        sm.add_transition(tensor, end,
            events=['parse'], input=']',
            action=self._end)
        
        sm.initialize()
        return sm

    def parse(self, tokens: str) -> typing.List[typing.Callable]:
        for token in tokens:
            self.sm.dispatch(pysm.Event('parse', input=token))
        return self.result
        
    def _list_only(self, state: pysm.State, event: pysm.Event) -> None:
        key = self.keys[self.counter]
        if not key:
            self.result.append(
                __ACTION_MAP__['NoOp']
            )
        else:
            self.result.append(functools.partial(
                __ACTION_MAP__['Tuple'], key=key
            ))

    def _single_out(self, state: pysm.State, event: pysm.Event) -> None:
        key = self.keys[self.counter]
        if not key:
            self.result.append(
                __ACTION_MAP__['NoOp']
            )
        else:
            self.result.append(functools.partial(
                __ACTION_MAP__['Tensor'], key=key
            ))

    def _end(self, state: pysm.State, event: pysm.Event) -> None:
        self.sm.stack.pop()

    def _advance(self, state: pysm.State, event: pysm.Event) -> None:
        self.counter += 1

    def _register(self, state: pysm.State, event: pysm.Event) -> None:
        # assert self.counter < len(self.keys),\
        # NOTE: '''
        #     Output tensor naming parsing stopped as an insufficient number of keys are present.
        #     Please check the output return type annotation and provide keys for all resulting tensors. 
        #     * Ignored tensor results can be omitted by using an empty string
        # '''
        key = self.keys[self.counter] if self.counter < len(self.keys) else ''
        if not key:
            if self.counter >= len(self.keys):
                log.warning(f"Less output keys were given ({self.keys}) than the operation supports."
                "This can either be due to an error or to ignore some outputs as they are not used."
                "In the latter case, the warning can be supressed using an empty string key \'\'.")
            self.result.append(
                __ACTION_MAP__['NoOp']
            )
        else:
            self.result.append(functools.partial(
                self.sm.stack.peek(), key=key
            ))

    def _start(self, state: pysm.State, event: pysm.Event) -> None:
        pass

    def _set_func(self, state: pysm.State, event: pysm.Event) -> None:
        self.sm.stack.push(__ACTION_MAP__[event.input])


def get_result_fillers(
    module: torch.nn.Module,
    keys: typing.Sequence[str]
):
    hints = str(typing.get_type_hints(module.forward)['return']) # return type annotations
    hints = hints.replace('typing.', '')\
        .replace('torch.','') # remove module names to keep only the pure type names
    tokens = re.split(r'(\W)', hints) # split in words
    tokens = list(toolz.remove(lambda s: not s or s.isspace(), tokens)) # remove empties
    rtp = ReturnTypeParser(keys=keys)
    fillers = rtp.parse(tokens) # get type specific filler functors
    
    def filler(
        tensor_dict: typing.Dict[str, torch.Tensor],
        out_tensor: torch.Tensor,
        fillers: typing.List[typing.Callable]
    ): # actual filler functor execution
        #TODO: check if out_tensor is pure tensor, then zip unwraps its batch dimension?
        for item, visitor in zip(
            [out_tensor] if (type(out_tensor) is list or type(out_tensor) is torch.Tensor) else out_tensor,
            itertools.cycle(fillers)
        ):
            visitor(tensor_dict, item)

    return functools.partial(filler, fillers=fillers) # return lambda with cached keys

def resolve_io_config(
    cfg: typing.Union[str, omegaconf.ListConfig]
):
    if type(cfg) is str:
        return [[cfg]]
    elif type(cfg) is omegaconf.ListConfig:
        return cfg if all(type(i) is omegaconf.ListConfig for i in cfg) else [cfg]

def split_as(
    frm: typing.List[typing.Any], 
    to: typing.List[typing.Any]
):
    if len(frm) == len(to):
        return frm
    elif len(frm) < len(to) and len(frm[0]) == len(to):
        return [[x] for x in frm[0]]

def force_list(i: typing.Union[str, typing.Any]):
    return [i] if type(i) is str else i