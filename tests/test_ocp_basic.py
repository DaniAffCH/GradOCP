import casadi as ca
import numpy as np
from GradOCP.core import GradOCP

def test_add_variable_and_stack():
    ocp = GradOCP()
    x = ocp.add_variable("x", 2, stage=0, lb=[0,0], ub=[1,1])
    p = ocp.add_parameter("p", 1)
    stacked_x = ocp._stack_variables()
    stacked_p = ocp._stack_parameters()
    assert stacked_x.shape == (2,1)
    assert stacked_p.shape == (1,1)

def test_add_constraints_and_stack():
    ocp = GradOCP()
    x = ocp.add_variable("x", 1, stage=0)
    c = ocp.add_constraint(x.symbol - 1, stage=0)
    stacked_g = ocp._stack_constraints()
    assert stacked_g.shape[0] == 1
