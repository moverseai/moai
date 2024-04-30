from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
# from hydra.plugins.plugin import Plugin
# from hydra.plugins.completion_plugin import CompletionPlugin
# from hydra.plugins.launcher import Launcher

from lark import Lark

import omegaconf.omegaconf

'''
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

'''
ADD_OP: "+" | "-" 
MUL_OP: "*" | "/"
POWER_OP: "^"
!add_expr: add_expr ADD_OP mul_expr
    | mul_expr
!mul_expr: mul_expr MUL_OP pow_expr
    | pow_expr
!pow_expr: primary POWER_OP pow_expr
    | primary
!primary: "-" NUMBER
    | NUMBER
    | "(" expr ")"
!expr: add_expr
'''

__MOAI_GRAMMAR__ = """

    ?name: FIELD ["." FIELD]
    ?names: name ("," name)*

    ADD_OP: "+" | "-" 
    MUL_OP: "*" | "/"
    POWER_OP: "^"
    
    ?add: sum "+" prod -> add
        | prod
    ?sub: sum "-" prod -> sub
        | prod
    ?sum: add | sub

    //?sum: sum ADD_OP prod
    //    | prod
    
    ?mul: prod "*" pow -> mul
        | pow
    ?div: prod "/" pow -> div
        | pow
    ?prod: mul | div
    //?prod: prod MUL_OP pow
    //    | pow    
    
    ?pow: primary POWER_OP pow
        | primary
    
    
    ?primary: "-" NUMBER
        | NUMBER                            -> number
        | name                              -> extract        
        | "cat" "(" names "," NUMBER ")"    -> cat
        | "stack" "(" names "," NUMBER ")"  -> stack
        | "(" expr ")"
    ?expr: sum

    %import common.CNAME -> FIELD
    %import common.NUMBER
    %import common.WS_INLINE
    %import common.WS

    %ignore WS
    %ignore WS_INLINE
"""

__MOAI_GRAMMAR_TEST__ = """
    ?name: FIELD ["." FIELD]
    ?names: name ("," name)*
    
    ?start: factor ( ( "-" | "+" ) factor )*
    
    ?sum: factor
        | factor "+" factor   -> add
        | factor "-" factor   -> sub
    
    ?factor: unary ( ( "/" | "*" ) unary )*
    ?unary: ( "!" | "-" ) unary | nnary
    ?nnary: "cat" "(" names "," NUMBER ")" -> cat
        | "stack" "(" names "," NUMBER ")" -> stack
        | primary
    ?primary: NUMBER -> number
        | "-" primary      -> neg
        | name             -> extract
        | "(" start ")"

    %import common.CNAME -> FIELD
    %import common.NUMBER
    %import common.WS_INLINE
    %import common.WS

    %ignore WS
    %ignore WS_INLINE
"""

__MOAI_GRAMMAR_OLD__ = """
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