""" Scikit decorators for optimizing hyperparameters """

from pySPACE.missions.nodes import DEFAULT_NODE_MAPPING, NODE_MAPPING
# noinspection PyUnresolvedReferences
from pySPACE.missions.nodes.scikit_nodes import SVRRegressorSklearnNode
from pySPACE.missions.nodes.decorators import LogUniformParameter, LogNormalParameter,\
    ChoiceParameter, QLogUniformParameter, NoOptimizationParameter
from pySPACE.missions.nodes import scikit_nodes


@LogUniformParameter("C", min_value=1e-6, max_value=1e6)
@LogNormalParameter("epsilon", shape=0.1 / 2, scale=0.1)
@ChoiceParameter("kernel",choices=["linear", "rbf", "poly", "sigmoid", "precomputed"])
#degree int, default: 3
@LogUniformParameter("gamma", min_value=1e-6, max_value=1e3)
# coef0: float, default: 0.0
@NoOptimizationParameter("shrinking")
#tol: float, default: 1e-3
@NoOptimizationParameter("cache_size")
@NoOptimizationParameter("verbose")
@QLogUniformParameter("max_iter", min_value=1, max_value=1e6, q=1)
class OptSVRRegressorSklearnNode(SVRRegressorSklearnNode):
    def __init__(self, C=1, epsilon=0.1, kernel="rbf", degree=3, gamma="auto", coef0=0.0, shrinking=True, tol=1e-3, 
                 verbose=False, max_iter=-1, **kwargs):
        super(OptSVRRegressorSklearnNode, self).__init__(C=C, epsilon=epsilon, kernel=kernel, degree=int(degree),
                                                         gamma=gamma, coef0=coef0, shrinking=shrinking, tol=tol,
                                                         verbose=verbose, max_iter=int(max_iter),
                                                         **kwargs)

try:
    from svext import SVR as IncSVR
    inc_svr = scikit_nodes.wrap_scikit_predictor(IncSVR)


    class OptIncSVRRegressorSklearnNode(inc_svr):
        def __init__(self, C=1, epsilon=0.1, kernel="rbf", degree=3, gamma="auto", coef0=0.0, shrinking=True, tol=1e-3,
                     verbose=False, max_iter=-1, **kwargs):
            super(OptIncSVRRegressorSklearnNode, self).__init__(C=C, epsilon=epsilon, kernel=kernel, degree=int(degree),
                                                                gamma=gamma, coef0=coef0, shrinking=shrinking, tol=tol,
                                                                verbose=verbose, max_iter=int(max_iter),
                                                                **kwargs)


    DEFAULT_NODE_MAPPING[inc_svr.__name__] = inc_svr
    NODE_MAPPING[inc_svr.__name__] = inc_svr
    NODE_MAPPING[inc_svr.__name__[:-4]] = inc_svr
except ImportError:
    pass

