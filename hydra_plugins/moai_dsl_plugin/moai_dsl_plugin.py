import omegaconf.omegaconf
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from lark import Lark

# from hydra.plugins.plugin import Plugin
# from hydra.plugins.completion_plugin import CompletionPlugin
# from hydra.plugins.launcher import Launcher


# NOTE: faster parsing: http://blog.erezsh.com/5-lark-features-you-probably-didnt-know-about/#5-lark-cython


""" EXAMPLE GRAMMAR USED AS BASE
expression     → equality ;
equality       → comparison ( ( "!=" | "==" ) comparison )* ;
comparison     → term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
term           → factor ( ( "-" | "+" ) factor )* ;
factor         → unary ( ( "/" | "*" ) unary )* ;
unary          → ( "!" | "-" ) unary
               | primary ;
primary        → NUMBER | STRING | "true" | "false" | "nil"
               | "(" expression ")" ;
"""

# TODO:  bmm, (mod?),
#       dot (symbol? ! or # or %? or | or ', or func, i.e. dot(x,y)),
#       einsum, matrix ops (inverse&transpose)
#       lin/logspace, (a)range, lerp
#       full(_like) [maybe obsolete cause of number math], unflatten (tricky),
#       sigmoid, softmax, mean, std, var,
#       cartesian product, cdist, covariance, cum(sum/prod),
#       flip, normalize, roll, multi_dot, norm,
#       angles/fused math/clamp/floor/ceil

__MOAI_GRAMMAR__ = """

    ?name: FIELD ("." FIELD)*
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

    ?index: MINUS1 | MINUS2 | MINUS3 | MINUS4 | SIGNED_INT
    ?indices: "[" INT ("," INT)* "]"
    ?slice: index? ":" index?
    // ?slice_to: ":" index
    // ?slice_from: index ":"
    ?slicing: ALL | ELLIPSIS | index | NEWAXIS | indices | slice // | slice_to | slice_from
        
    ?primary: "-" NUMBER
        | "-" name                          -> neg
        | "-" expr                          -> neg
        | "exp" "(" name ")"                -> exp
        | "exp" "(" expr ")"                -> exp
        | "log" "(" name ")"                -> log
        | "log" "(" expr ")"                -> log
        | "abs" "(" name ")"                -> abs
        | "abs" "(" expr ")"                -> abs
        | "cos" "(" name ")"                -> cos
        | "cos" "(" expr ")"                -> cos        
        | "acos" "(" name ")"               -> acos
        | "acos" "(" expr ")"               -> acos
        | "sin" "(" name ")"                -> sin
        | "sin" "(" expr ")"                -> sin
        | "asin" "(" name ")"               -> asin
        | "asin" "(" expr ")"               -> asin
        | "tan" "(" name ")"                -> tan
        | "tan" "(" expr ")"                -> tan
        | "tanh" "(" name ")"               -> tanh
        | "tanh" "(" expr ")"               -> tanh
        | "atan" "(" name ")"               -> atan
        | "atan" "(" expr ")"               -> atan
        | "sqrt" "(" name ")"               -> sqrt
        | "sqrt" "(" expr ")"               -> sqrt
        | "deg" "(" name ")"                -> rad2deg
        | "deg" "(" expr ")"                -> rad2deg
        | "rad" "(" name ")"                -> deg2rad
        | "rad" "(" expr ")"                -> deg2rad
        | "reciprocal" "(" name ")"         -> reciprocal
        | "reciprocal" "(" expr ")"         -> reciprocal
        | NUMBER                            -> number        
        | name                              -> extract        
        | "cat" "(" names "," SIGNED_INT ")"    -> cat
        | "stack" "(" names "," SIGNED_INT ")"  -> stack        
        | "view" "(" name "," SIGNED_INT ("," SIGNED_INT)* ")"  -> reshape
        | "transpose" "(" name "," SIGNED_INT ("," SIGNED_INT)* ")"  -> transpose
        | "flatten" "(" name "," SIGNED_INT ["," SIGNED_INT] ")"  -> flatten
        | "repeat_interleave" "(" name "," SIGNED_INT "," SIGNED_INT ")" -> repeat
        | "roll" "(" name "," SIGNED_INT "," SIGNED_INT ")" -> roll
        | "zeros" "(" name ")"              -> zeros_like
        | "ones" "(" name ")"               -> ones_like
        | "rand" "(" name ")"               -> rand_like
        | "randn" "(" name ")"              -> randn_like
        | "unsq" "(" name "," SIGNED_INT ("," SIGNED_INT)* ")" -> unsqueeze
        | "sq" "(" name ("," SIGNED_INT)* ")" -> squeeze
        | "expand_batch_as" "(" name "," name ")" -> expand_batch_as
        | "reshape_as" "(" name "," name ")" -> reshape_as
        | name "[" slicing ("," slicing)* "]" -> slicing
        | "(" expr ")"
    
    ?expr: sum

    ALL: ":"
    ELLIPSIS: "..."
    NEWAXIS: "new"
    MINUS1: "-1"
    MINUS2: "-2"
    MINUS3: "-3"
    MINUS4: "-4"

    %import common.CNAME -> FIELD
    %import common.NUMBER
    %import common.SIGNED_INT
    %import common.INT
    %import common.WS_INLINE
    %import common.WS

    %ignore WS
    %ignore WS_INLINE
"""


class MoaiDSLPlugin(SearchPathPlugin):
    def __init__(self) -> None:
        self.parser = Lark(__MOAI_GRAMMAR__, parser="earley", start="expr")
        if not omegaconf.OmegaConf.has_resolver("mi"):
            omegaconf.OmegaConf.register_new_resolver("mi", self._parse_expression)

    def _parse_expression(self, *expressions):
        text = ",".join(expressions)
        return self.parser.parse(text)

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        pass  # NOTE: only used to ensure proper resolver registration as searchpath is called first
