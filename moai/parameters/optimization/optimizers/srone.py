from typing import Iterable, Optional
try:
    from pytorch_soo.quasi_newton import QuasiNewton, QuasiNewtonTrust, TrustRegionSpec
    import pytorch_soo.matrix_free_operators as matrix_ops
    from pytorch_soo.line_search_spec import LineSearchSpec
    from pytorch_soo.solvers import ConjugateResidual
except ImportError:
    print("Please install pytorch_soo(https://github.com/pnnl/pytorch_soo) to use this module.")
import toolz

class SymmetricRankOne(QuasiNewton):
    """
    https://en.wikipedia.org/wiki/Symmetric_rank-one
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        if isinstance(toolz.first(params), dict) and len(params) > 1:
            params = list(toolz.mapcat(lambda g: list(g),
                toolz.mapcat(
                    lambda d: list(v for v in d.values()), 
                params)
            ))
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = matrix_ops.SymmetricRankOne(lambda p: p, n=matrix_free_memory)
        self.solver = ConjugateResidual(max_krylov, krylov_tol)

class SymmetricRankOneTrust(QuasiNewtonTrust):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            trust_region=trust_region,
        )
        self.mf_op = matrix_ops.SymmetricRankOne(lambda p: p, n=matrix_free_memory)