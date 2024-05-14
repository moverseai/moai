from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
# from hydra.plugins.plugin import Plugin
# from hydra.plugins.completion_plugin import CompletionPlugin
# from hydra.plugins.launcher import Launcher

from lark import Lark
#NOTE: faster parsing: http://blog.erezsh.com/5-lark-features-you-probably-didnt-know-about/#5-lark-cython

import omegaconf.omegaconf

''' EXAMPLE GRAMMAR USED AS BASE
expression     → equality ;
equality       → comparison ( ( "!=" | "==" ) comparison )* ;
comparison     → term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
term           → factor ( ( "-" | "+" ) factor )* ;
factor         → unary ( ( "/" | "*" ) unary )* ;
unary          → ( "!" | "-" ) unary
               | primary ;
primary        → NUMBER | STRING | "true" | "false" | "nil"
               | "(" expression ")" ;
'''

#TODO: slicing, bmm, (mod?), dot (symbol? ! or # or %? or | or ', or func, i.e. dot(x,y))

__MOAI_GRAMMAR__ = """

    ?name: FIELD ["." FIELD]
    ?names: name ("," name)*
    
    ?add: sum "+" prod -> add
        | sum "+" gen -> addg
        | gen "+" sum -> addg
        | prod
    ?sub: sum "-" prod -> sub
        | sum "-" gen -> subg
        | gen "-" sum -> subg
        | prod
    ?sum: add | sub
    
    ?mul: prod "*" pow -> mul
        | prod "*" gen -> mulg
        | gen "*" prod -> mulg
        | pow
    ?div: prod "/" pow -> div
        | prod "/" gen -> divg
        | gen "/" prod -> divg
        | pow
    ?prod: mul | div 
    
    ?pow: primary "^" pow -> pow
        | primary
    
    ?gen: "ones" "(" NUMBER ("," NUMBER)* ")" -> ones
        | "zeros" "(" NUMBER ("," NUMBER)* ")" -> zeros
        | "rand" "(" NUMBER ("," NUMBER)* ")" -> rand
        | "randn" "(" NUMBER ("," NUMBER)* ")" -> randn
    
    ?primary: "-" NUMBER
        | "-" name                          -> neg
        | "-" expr                          -> neg
        | NUMBER                            -> number        
        | name                              -> extract        
        | "cat" "(" names "," NUMBER ")"    -> cat
        | "stack" "(" names "," NUMBER ")"  -> stack        
        | "view" "(" name "," NUMBER ("," NUMBER)* ")"  -> reshape
        | "zeros" "(" name ")"              -> zeros_like
        | "ones" "(" name ")"               -> ones_like
        | "rand" "(" name ")"               -> rand_like
        | "randn" "(" name ")"              -> randn_like
        | "unsq" "(" name "," NUMBER ("," NUMBER)* ")" -> unsqueeze
        | "sq" "(" name ("," NUMBER)* ")" -> squeeze
        | "(" expr ")"
    ?expr: sum

    %import common.CNAME -> FIELD
    %import common.NUMBER
    %import common.WS_INLINE
    %import common.WS

    %ignore WS
    %ignore WS_INLINE
"""

class MoaiDSLPlugin(SearchPathPlugin):
    def __init__(self) -> None:
        self.parser = Lark(__MOAI_GRAMMAR__, parser='earley', start='expr')
        omegaconf.OmegaConf.register_new_resolver("mi", self._parse_expression)
    
    def _parse_expression(self, *expressions):
        text = ','.join(expressions)
        return self.parser.parse(text)

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        pass #NOTE: only used to ensure proper resolver registration as searchpath is called first