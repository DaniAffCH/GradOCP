import numpy as np
from GradOCP.core import GradOCP

def test_backward_gradient_shapes():
    ocp = GradOCP()
    x = ocp.add_variable("x", 1, stage=0, lb=[0], ub=[5])
    y = ocp.add_parameter("y", size=1)
    ocp.set_objective((x.symbol - y.symbol)**2)
    ocp.build(solver_opts={"ipopt.print_level":0, "print_time":0, "ipopt.sb":"yes"})
    ocp.solve(p_vals={"y": 2})

    grad = ocp.backward(inversion_method="pseudo")
    assert grad.shape[0] == 1
