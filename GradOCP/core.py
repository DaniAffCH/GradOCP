from typing import List, Optional, Dict, Any
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import casadi as ca
from GradOCP.variables import DecisionVariable, Parameter, Constraint

class GradOCP:
    def __init__(self):
        self.variables: List[DecisionVariable] = []
        self.parameters: List[Parameter] = []
        self.constraints: List[Constraint] = []
        self.obj: Any = ca.SX(0)
        self._built = False
        self._converged = False
        self._solver = None
        self._normal_constraints = []
        self._x = None
        self._p = None
        self._g = None
        self._eps_eps_J_func = None
        self._eps_J_func = None
        self._theta_eps_J_func = None
        self._lbx: List[float] = []
        self._ubx: List[float] = []
        self._lbg: List[float] = []
        self._ubg: List[float] = []
        self._var_slices: Dict[str, slice] = {}

    def add_variable(self, name, size, stage, lb=None, ub=None):
        var = DecisionVariable(name, size, stage, lb, ub)
        self.variables.append(var)
        return var
    
    def add_parameter(self, name: str, size: int = 1) -> Parameter:
        p = Parameter(name, size)
        self.parameters.append(p)
        return p

    def add_constraint(self, expr, stage, lb=0.0, ub=0.0):
        c = Constraint(expr, stage, lb, ub)
        self.constraints.append(c)
        return c
    
    def set_objective(self, expr):
        self.obj = expr

    def _stack_variables(self):
        if not self.variables:
            return ca.SX.zeros(0, 1)
        return ca.vertcat(*[v.symbol.reshape((-1, 1)) for v in self.variables])

    def _stack_parameters(self):
        if not self.parameters:
            return ca.SX.zeros(0, 1)
        return ca.vertcat(*[p.symbol.reshape((-1, 1)) for p in self.parameters])

    def _stack_constraints(self):
        if not self.constraints:
            return ca.SX.zeros(0, 1)
        return ca.vertcat(*[c.expr for c in self.constraints])
    
    def _compute_bounds(self):
        lbx, ubx = [], []
        offset = 0
        for v in self.variables:
            self._var_slices[v.name] = slice(offset, offset + v.size)
            offset += v.size
            lbx.extend([float(x) for x in v.lb])
            ubx.extend([float(x) for x in v.ub])
        lbg = [float(c.lb) for c in self.constraints]
        ubg = [float(c.ub) for c in self.constraints]
        self._lbx, self._ubx, self._lbg, self._ubg = lbx, ubx, lbg, ubg

    def _build_active_constraints(self, lam_g: np.ndarray, lam_x: np.ndarray):
        stage_dict = OrderedDict()
        offset_lam_x = 0

        for v in self.variables:
            g_stage = []
            for j, (lb, ub) in enumerate(zip(v.lb, v.ub)):
                if lb > -ca.inf and lam_x[offset_lam_x + j] < -1e-8:
                    expr = lb - v[j]
                    lam = -lam_x[offset_lam_x + j]
                    g_stage.append((expr, lam))
                elif ub < ca.inf and lam_x[offset_lam_x + j] > 1e-8:
                    expr = v[j] - ub
                    lam = lam_x[offset_lam_x + j]
                    g_stage.append((expr, lam))

            offset_lam_x += v.size
            if g_stage:
                stage_dict.setdefault(v.stage, {"g": [], "h": []})["g"].extend(g_stage)

        for idx, c in enumerate(self.constraints):
            stage_dict.setdefault(c.stage, {"g": [], "h": []})
            if c.is_equality():
                stage_dict[c.stage]["h"].append((c.expr, lam_g[idx]))
            else:
                if lam_g[idx] > 1e-8:
                    expr = c - c.ub
                    lam = lam_g[idx]
                    stage_dict[c.stage]["g"].append( (expr, lam) )
                elif lam_g[idx] < -1e-8:
                    expr = c.lb - c
                    lam = -lam_g[idx]
                    stage_dict[c.stage]["g"].append( (expr, lam) )

        for s in sorted(stage_dict.keys()):
            stage_dict.move_to_end(s)

        all_constraints = []
        all_lambdas = []
        for s in stage_dict.keys():
            stage = stage_dict[s]
            for expr, lam in stage["g"]:
                all_constraints.append(expr)
                all_lambdas.append(lam)
            for expr, lam in stage["h"]:
                all_constraints.append(expr)
                all_lambdas.append(lam)

        r = ca.vertcat(*all_constraints) if all_constraints else ca.SX.zeros(0, 1)
        r_lambda = ca.vertcat(*all_lambdas) if all_lambdas else ca.SX.zeros(0, 1)
        return r, r_lambda

    def build(self, solver_name: str = "ipopt", solver_opts: Optional[Dict] = None):
        x = self._stack_variables()
        p = self._stack_parameters()
        g = self._stack_constraints()
        self._compute_bounds()
        nlp = {"x": x, "p": p, "f": self.obj, "g": g}
        opts = {} if solver_opts is None else solver_opts
        self._solver = ca.nlpsol("nlp_solver", solver_name, nlp, opts)
        self._x, self._p, self._g = x, p, g

        H_J, grad_J = ca.hessian(self.obj, x)
        self._eps_eps_J_func = ca.Function("eps_eps_J", [x, p], [H_J])
        self._eps_J_func = ca.Function("eps_J", [x, p], [grad_J])
        H_theta_J = ca.jacobian(grad_J, p)
        self._theta_eps_J_func = ca.Function("theta_eps_J", [x,p], [H_theta_J])

        self._built = True

    def _pack_p(self, p_vals: Optional[Dict[str, Any]]) -> np.ndarray:
        if not self.parameters:
            return np.array([])
        if p_vals is None:
            return np.zeros(sum(p.size for p in self.parameters))
        flat = []
        for p in self.parameters:
            v = p_vals.get(p.name)
            if v is None:
                raise KeyError(p.name)
            arr = np.asarray(v).ravel()
            if arr.size != p.size:
                raise ValueError(p.name)
            flat.extend(arr.tolist())
        return np.asarray(flat)
        
    def solve(self, p_vals: Optional[Dict[str, Any]] = None, x0: Optional[np.ndarray] = None, solver_opts_override: Optional[Dict] = None, lbx: Optional[List[float]] = None, ubx: Optional[List[float]] = None, lbg: Optional[List[float]] = None, ubg: Optional[List[float]] = None) -> Dict[str, Any]:
        if not self._built:
            self.build()
        p_vec = self._pack_p(p_vals)
        arg = {"p": p_vec}
        if x0 is not None:
            arg["x0"] = x0
        arg["lbx"] = self._lbx if lbx is None else lbx
        arg["ubx"] = self._ubx if ubx is None else ubx
        arg["lbg"] = self._lbg if lbg is None else lbg
        arg["ubg"] = self._ubg if ubg is None else ubg
        if solver_opts_override:
            self._solver = ca.nlpsol("nlp_solver", "ipopt", {"x": self._x, "p": self._p, "f": self.obj, "g": self._g}, solver_opts_override)
        res = self._solver(**arg)

        self._converged = self._solver.stats()["return_status"] == "Solve_Succeeded"

        x_opt = np.array(res["x"].full()).ravel()
        lam_g = np.array(res["lam_g"].full()).ravel() if "lam_g" in res else None
        lam_x = np.array(res["lam_x"].full()).ravel() if "lam_x" in res else None
        self._last_solution = {"x": self.unpack_x(x_opt), "x_raw": x_opt, "lam_g": lam_g, "lam_x": lam_x, "p": p_vec, "J": res["f"]}

        return deepcopy(self._last_solution)

    def unpack_x(self, x_vec: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        for v in self.variables:
            s = self._var_slices[v.name]
            out[v.name] = x_vec[s].reshape((v.size,))
        return out
    
    # TODO: use VJP instead
    def backward(self, inversion_method = "pseudo") -> np.ndarray:
        if not self._built:
            raise RuntimeError("OCP must be built before computing derivatives")
        if not hasattr(self, "_last_solution"):
            raise RuntimeError("Solve the problem first before calling backward()")
        if not self._converged:
            raise RuntimeError("Cannot compute backward derivatives: the last solve did not converge to a KKT-satisfying point.")


        x_val = self._last_solution["x_raw"]
        p_val = self._last_solution["p"]
        lam_g = self._last_solution["lam_g"]
        lam_x = self._last_solution["lam_x"]

        active_constr, lambda_constr = self._build_active_constraints(lam_g, lam_x)

        A_sym = ca.jacobian(active_constr, self._x)
        eps_r_func = ca.Function("eps_r", [self._x, self._p], [A_sym])
        A = eps_r_func(x_val, p_val)

        C_sym = ca.jacobian(active_constr, self._p)
        theta_r_func = ca.Function("theta_r", [self._x, self._p], [C_sym])
        C = theta_r_func(x_val, p_val)

        sh = ca.DM.zeros(self._x.shape[0], self._x.shape[0])
        sb = ca.DM.zeros(self._x.shape[0], self._p.shape[0])

        # TODO: this can be vectorized
        for i in range(A.shape[0]):
            eps_eps_r_sym = ca.jacobian(A[i,:], self._x)
            theta_eps_r_sym = ca.jacobian(A[i,:], self._p)

            eps_eps_r_func = ca.Function("eps_eps_r", [self._x, self._p], [eps_eps_r_sym])
            theta_eps_r_func = ca.Function("theta_eps_r", [self._x, self._p], [theta_eps_r_sym])

            eps_eps_r = eps_eps_r_func(x_val, p_val)
            theta_eps_r = theta_eps_r_func(x_val, p_val)

            sh += lambda_constr[i] * eps_eps_r
            sb += lambda_constr[i] * theta_eps_r

        H = self._eps_eps_J_func(x_val, p_val) - sh
        B = self._theta_eps_J_func(x_val, p_val) - sb

        inversion_methods = {
            "exact": np.linalg.inv,
            "pseudo": np.linalg.pinv,
            "regularization": lambda x: x + np.eye(x.shape[0]) * 1e-3,
        }

        inversion_function = inversion_methods.get(inversion_method)
        if inversion_function is None:
            raise ValueError(f"Unknown inversion method: {inversion_method}")

        H_inv = inversion_function(H)
        H_inv_B = H_inv @ B
        H_inv_At = H_inv @ A.T
        M = A @ H_inv_At
        rhs = A @ H_inv_B - C
        return H_inv_At @ inversion_function(M) @ rhs - H_inv_B