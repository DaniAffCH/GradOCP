import casadi as ca
import numpy as np
import pytest
from GradOCP.core import GradOCP

def test_bilevel_convergence():
    ocp = GradOCP()

    x = ocp.add_variable("x_0", size=1, lb=[0], ub=[400], stage=0)
    y = ocp.add_parameter("y", size=1)

    N = 5
    y_param = 0.1
    lr = 0.001
    for k in range(N-1):
        x_next = ocp.add_variable(f"x_{k+1}", size=1, lb=[0], ub=[400], stage=k+1)
        ocp.add_constraint(x_next - x - y, stage=k)
        x = x_next

    ocp.set_objective((x_next - 3)**2 + (y - 1)**2)

    ocp.build(solver_opts={"ipopt.print_level":0, "print_time":0, "ipopt.sb":"yes"})

    errors = []
    for i in range(30):
        res = ocp.solve(p_vals={"y": y_param})
        xT = res["x"][f"x_{N-1}"].item()
        grad_x = np.zeros(N)
        grad_x[-1] = 2*(xT - 3)
        grad_y = 2*(y_param - 1)

        dx_dy = ocp.backward(inversion_method="regularization")
        dJ_dy = dx_dy.T @ grad_x + grad_y

        y_param -= lr * dJ_dy
        errors.append(float(res["J"]))

    # Assert the objective decreased
    assert errors[-1] < errors[0], f"Objective did not decrease: start={errors[0]}, end={errors[-1]}"

def test_bilevel_infeasible():
    ocp = GradOCP()

    x = ocp.add_variable("x_0", size=1, lb=[0], ub=[400], stage=0)
    y = ocp.add_parameter("y", size=1)

    N = 5
    y_param = 1e6  # huge value to trigger infeasibility
    for k in range(N-1):
        x_next = ocp.add_variable(f"x_{k+1}", size=1, lb=[0], ub=[400], stage=k+1)
        ocp.add_constraint(x_next - x - y, stage=k)
        x = x_next

    ocp.set_objective((x_next - 3)**2 + (y - 1)**2)
    ocp.build(solver_opts={"ipopt.print_level":0, "print_time":0, "ipopt.sb":"yes"})

    with pytest.raises(RuntimeError):
        res = ocp.solve(p_vals={"y": y_param})
        if not ocp._converged:
            raise RuntimeError("Solver did not converge (infeasible)")
