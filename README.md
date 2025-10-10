# GradOCP

Python framework for bilevel optimization using CasADi and [IDOC](https://proceedings.neurips.cc/paper_files/paper/2023/file/bcfcf7232cb74e1ef82d751880ff835b-Paper-Conference.pdf). Solve optimal control problems, backpropagate through their solutions, and optimize parameters.

---

## Example: Bilevel optimization with GradOCP

```python
import casadi as ca
import numpy as np
from GradOCP.core import GradOCP

# Create a bilevel optimal control problem
ocp = GradOCP()

# Variables and parameters
x = ocp.add_variable("x_0", size=1, lb=[0], ub=[400], stage=0)
y = ocp.add_parameter("y", size=1)

N = 5  # Horizon length
y_param = 0.1  # Initial parameter
lr = 0.001  # Learning rate

# Define system dynamics: x_{k+1} = x_k + y
for k in range(N - 1):
    x_next = ocp.add_variable(f"x_{k+1}", size=1, lb=[0], ub=[400], stage=k + 1)
    ocp.add_constraint(x_next - x - y, stage=k)
    x = x_next

# Terminal objective
ocp.set_objective((x_next - 3)**2 + (y - 1)**2)

# Build the problem
ocp.build()

# Outer-loop optimization (bilevel)
for i in range(30):
    res = ocp.solve(p_vals={"y": y_param})
    xT = res["x"][f"x_{N-1}"].item()

    grad_x = np.zeros(N)
    grad_x[-1] = 2 * (xT - 3)
    grad_y = 2 * (y_param - 1)

    dx_dy = ocp.backward(inversion_method="regularization")
    dJ_dy = dx_dy.T @ grad_x + grad_y

    # Gradient descent on y
    y_param -= lr * dJ_dy

print(f"Optimized y ≈ {y_param:.3f}")
```
