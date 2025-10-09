import numpy as np
from GradOCP.core import GradOCP

def test_simple_ocp_solve():
    ocp = GradOCP()
    x_var = ocp.add_variable("x", 1, stage=0, lb=[0], ub=[30])
    y_param = ocp.add_parameter("y", size=1)
    ocp.add_constraint(x_var - y_param, stage=0)
    ocp.set_objective((x_var.symbol - 3)**2 + (y_param.symbol - 1)**2)
    ocp.build(solver_opts={"ipopt.print_level":0, "print_time":0, "ipopt.sb":"yes"})
    
    
    for y_val in [2, 5, 10, 20]:
        res = ocp.solve(p_vals={"y": y_val})
        assert ocp._converged
        x_val = res["x"]["x"].item()
        assert np.isclose(x_val, y_val)
