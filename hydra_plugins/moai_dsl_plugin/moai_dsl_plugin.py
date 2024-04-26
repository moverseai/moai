from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
# from hydra.plugins.plugin import Plugin
# from hydra.plugins.completion_plugin import CompletionPlugin
# from hydra.plugins.launcher import Launcher

from lark import Lark

import omegaconf.omegaconf

__MOAI_GRAMMAR__ = """
    ?start: sum
          | name "=" sum    -> assign_var

    ?name: FIELD ["." FIELD]
    ?names: name ("," name)*

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: atom
        | product "*" atom  -> mul
        | product "/" atom  -> div

    ?atom: NUMBER           -> number
         | "-" atom         -> neg
         | name             -> extract         
         | "cat" "(" names "," NUMBER ")" -> cat
         | "stack" "(" names "," NUMBER ")" -> stack
         | "(" sum ")"

    %import common.CNAME -> FIELD
    %import common.NUMBER
    %import common.WS_INLINE
    %import common.WS

    %ignore WS
    %ignore WS_INLINE
"""

class MoaiDSLPlugin(SearchPathPlugin):
    def __init__(self) -> None:
        self.parser = Lark(__MOAI_GRAMMAR__, parser='earley')
        omegaconf.OmegaConf.register_new_resolver("mi", self._parse_expression)
    
    def _parse_expression(self, *expressions):
        text = ','.join(expressions)
        return self.parser.parse(text)

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        pass #NOTE: only used to ensure proper resolver registration as searchpath is called first