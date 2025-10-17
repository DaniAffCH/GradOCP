import casadi as ca
import numpy as np
from copy import deepcopy
from typing import List, Optional, Dict, Any, Sequence, Union
import pprint

SXOrMX = Union[ca.SX, ca.MX]
CasadiFunction = ca.Function
CASADI_RETURN_OK = "Solve_Succeeded"
class GradOCP:
    '''
    The dynamics is the only component not expressed in canonical form. This formulation allows expressing it with respect to the current state, yielding the state at time t+1. 
    This property is explicitly handled during sensitivity computation.
    '''
    dynamics: Optional[SXOrMX]
    dynamics_fn: Optional[CasadiFunction]
    _dynamics_jac_traj: Optional[SXOrMX]
    _dynamics_jac_traj_fn: Optional[CasadiFunction]
    _dynamics_jac_params: Optional[SXOrMX]
    _dynamics_jac_params_fn: Optional[CasadiFunction]
    
    state: Optional[SXOrMX]
    control: Optional[SXOrMX]
    parameters: Optional[SXOrMX]

    _stage_ineq_constr: List[SXOrMX]
    _stage_ineq_constr_fn: List[CasadiFunction]
    _stage_ineq_constr_jac_traj: List[SXOrMX]
    _stage_ineq_constr_jac_traj_fn: List[CasadiFunction]
    _stage_ineq_constr_jac_params: List[SXOrMX]
    _stage_ineq_constr_jac_params_fn: List[CasadiFunction]
    
    _stage_eq_constr: List[SXOrMX]
    _stage_eq_constr_fn: List[CasadiFunction]
    _stage_eq_constr_jac_traj: List[SXOrMX]
    _stage_eq_constr_jac_traj_fn: List[CasadiFunction]
    _stage_eq_constr_jac_params: List[SXOrMX]
    _stage_eq_constr_jac_params_fn: List[CasadiFunction]

    _terminal_ineq_constr: List[SXOrMX]
    _terminal_ineq_constr_fn: List[CasadiFunction]
    _terminal_ineq_constr_jac_state: List[SXOrMX]
    _terminal_ineq_constr_jac_state_fn: List[CasadiFunction]
    _terminal_ineq_constr_jac_params: List[SXOrMX]
    _terminal_ineq_constr_jac_params_fn: List[CasadiFunction]

    _terminal_eq_constr: List[SXOrMX]
    _terminal_eq_constr_fn: List[CasadiFunction]
    _terminal_eq_constr_jac_state: List[SXOrMX]
    _terminal_eq_constr_jac_state_fn: List[CasadiFunction]
    _terminal_eq_constr_jac_params: List[SXOrMX]
    _terminal_eq_constr_jac_params_fn: List[CasadiFunction]

    _state_lb: List[float]
    _state_ub: List[float]

    _control_lb: List[float]
    _control_ub: List[float]

    _stage_cost: Optional[SXOrMX]
    _stage_cost_fn: Optional[CasadiFunction]

    _terminal_cost: Optional[SXOrMX]
    _terminal_cost_fn: Optional[CasadiFunction]
    
    _stage_L: Optional[SXOrMX]
    _stage_L_H: Optional[SXOrMX]
    _stage_L_H_fn: Optional[CasadiFunction]
    _stage_L_H_p: Optional[SXOrMX]
    _stage_L_H_p_fn: Optional[CasadiFunction]
    
    _terminal_L: Optional[SXOrMX]
    _terminal_L_H: Optional[SXOrMX]
    _terminal_L_H_fn: Optional[CasadiFunction]
    _terminal_L_H_p: Optional[SXOrMX]
    _terminal_L_H_p_fn: Optional[CasadiFunction]
    
    # Casadi problem
    _lbw: List[float]
    _ubw: List[float]
    _lbg: List[float]
    _ubg: List[float]
    _g: List[SXOrMX]
    _w: List[SXOrMX]
    _w0: List[float]
    _obj: Union[float, SXOrMX]
    
    _converged: bool
    _last_horizon: int
    tol: float
    
    _solver: CasadiFunction

    def __init__(self) -> None:
        self.dynamics = None
        self.dynamics_fn = None
        self._dynamics_jac_traj = None
        self._dynamics_jac_traj_fn = None
        self._dynamics_jac_params = None
        self._dynamics_jac_params_fn = None

        self.state = None
        self.control = None
        self.parameters = None
        
        self._stage_ineq_constr = []
        self._stage_ineq_constr_fn = []
        self._stage_ineq_constr_jac_traj = []
        self._stage_ineq_constr_jac_traj_fn = []
        self._stage_ineq_constr_jac_params = []
        self._stage_ineq_constr_jac_params_fn = []
        
        self._stage_eq_constr = []
        self._stage_eq_constr_fn = []
        self._stage_eq_constr_jac_traj = []
        self._stage_eq_constr_jac_traj_fn = []
        self._stage_eq_constr_jac_params = []
        self._stage_eq_constr_jac_params_fn = []

        self._terminal_ineq_constr = []
        self._terminal_ineq_constr_fn = []
        self._terminal_ineq_constr_jac_state = []
        self._terminal_ineq_constr_jac_state_fn = []
        self._terminal_ineq_constr_jac_params = []
        self._terminal_ineq_constr_jac_params_fn = []
        
        self._terminal_eq_constr = []
        self._terminal_eq_constr_fn = []
        self._terminal_eq_constr_jac_state = []
        self._terminal_eq_constr_jac_state_fn = []
        self._terminal_eq_constr_jac_params = []
        self._terminal_eq_constr_jac_params_fn = []

        self._state_lb = []
        self._state_ub = []

        self._control_lb = []
        self._control_ub = []

        self._stage_cost = None
        self._stage_cost_fn = None

        self._terminal_cost = None
        self._terminal_cost_fn = None
        
        self._stage_L = None
        self._stage_L_H = None
        self._stage_L_H_fn = None
        self._stage_L_H_p = None
        self._stage_L_H_p_fn = None
        
        self._terminal_L = None
        self._terminal_L_H = None
        self._terminal_L_H_fn = None
        self._terminal_L_H_p = None
        self._terminal_L_H_p_fn = None

        # Casadi problem
        self._lbw = []
        self._ubw = []
        self._lbg = []
        self._ubg = []
        self._g = []
        self._w = []
        self._w0 = []
        self._obj = 0
        
        self._solver = None
        self._converged = False
        self._last_horizon = None

        self.tol = 1e-8

    def add_state(self, state: SXOrMX, lb: Optional[Sequence[float]] = None, ub: Optional[Sequence[float]] = None) -> None:
        if not isinstance(state, (ca.MX, ca.SX)):
            raise TypeError("state must be a CasADi symbolic array")

        self.state = state
        n = state.shape[0]

        if lb is not None:
            lb_list = np.atleast_1d(lb).tolist()
            if len(lb_list) != n:
                raise ValueError("lb must have the same length as state")
            self._state_lb = lb_list
        else:
            self._state_lb = [-ca.inf] * n

        if ub is not None:
            ub_list = np.atleast_1d(ub).tolist()
            if len(ub_list) != n:
                raise ValueError("ub must have the same length as state")
            self._state_ub = ub_list
        else:
            self._state_ub = [ca.inf] * n

    def add_control(self, control: SXOrMX, lb: Optional[Sequence[float]] = None, ub: Optional[Sequence[float]] = None) -> None:
        if not isinstance(control, (ca.MX, ca.SX)):
            raise TypeError("control must be a CasADi symbolic array")

        self.control = control
        n = control.shape[0]

        if lb is not None:
            lb_list = np.atleast_1d(lb).tolist()
            if len(lb_list) != n:
                raise ValueError("lb must have the same length as control")
            self._control_lb = lb_list
        else:
            self._control_lb = [-ca.inf] * n

        if ub is not None:
            ub_list = np.atleast_1d(ub).tolist()
            if len(ub_list) != n:
                raise ValueError("ub must have the same length as control")
            self._control_ub = ub_list
        else:
            self._control_ub = [ca.inf] * n

    def add_parameters(self, parameter: SXOrMX) -> None:
        if not isinstance(parameter, (ca.MX, ca.SX)):
            raise TypeError("parameters must be a CasADi symbolic array")
        self.parameters = parameter

    def add_dynamics(self, dynamic: SXOrMX) -> None:
        if not all(v is not None for v in (self.state, self.control, self.parameters)):
            raise ValueError("state, control, and parameters must be set before adding dynamics")

        if isinstance(dynamic, (ca.MX, ca.SX)):
            self.dynamics = dynamic
            self.dynamics_fn = ca.Function('dynamics', [self.state, self.control, self.parameters], [self.dynamics])
        else:
            raise TypeError("dynamic must be a CasADi MX or SX expression")

    def add_stage_constraint(self, constraint: SXOrMX, lb=None, ub=None) -> None:
        if not isinstance(constraint, (ca.MX, ca.SX)):
            raise TypeError("constraint must be a CasADi symbolic expression")

        n = constraint.shape[0]
        ub_list = np.atleast_1d(ub).tolist() if ub is not None else None
        lb_list = np.atleast_1d(lb).tolist() if lb is not None else None

        if ub_list is not None and len(ub_list) != n:
            raise ValueError("ub must have the same length as constraint")
        if lb_list is not None and len(lb_list) != n:
            raise ValueError("lb must have the same length as constraint")

        eq_terms = []
        ineq_terms = []

        if ub_list is not None and lb_list is not None and all(abs(u - l) < self.tol for u, l in zip(ub_list, lb_list)):
            for i in range(n):
                eq_terms.append(constraint[i] - ub_list[i])
        else:
            if ub_list is not None:
                for i in range(n):
                    ineq_terms.append(constraint[i] - ub_list[i])
            if lb_list is not None:
                for i in range(n):
                    ineq_terms.append(lb_list[i] - constraint[i])

        if eq_terms:
            eq_vec = ca.vertcat(*eq_terms)
            self._stage_eq_constr.append(eq_vec)
            self._stage_eq_constr_fn.append(
                ca.Function(f"sec_{len(self._stage_eq_constr)}", [self.state, self.control, self.parameters], [eq_vec])
            )

        if ineq_terms:
            ineq_vec = ca.vertcat(*ineq_terms)
            self._stage_ineq_constr.append(ineq_vec)
            self._stage_ineq_constr_fn.append(
                ca.Function(f"sic_{len(self._stage_ineq_constr)}", [self.state, self.control, self.parameters], [ineq_vec])
            )

    def add_stage_cost(self, cost: SXOrMX) -> None:
        if not isinstance(cost, (ca.MX, ca.SX)):
            raise TypeError("stage cost must be a CasADi symbolic expression")
        self._stage_cost = cost
        self._stage_cost_fn = ca.Function("stage_cost", [self.state, self.control, self.parameters], [self._stage_cost])

    def add_terminal_constraint(self, constraint: SXOrMX, lb=None, ub=None) -> None:
        if not isinstance(constraint, (ca.MX, ca.SX)):
            raise TypeError("constraint must be a CasADi symbolic expression")

        n = constraint.numel()
        ub_list = np.atleast_1d(ub).tolist() if ub is not None else None
        lb_list = np.atleast_1d(lb).tolist() if lb is not None else None

        if ub_list is not None and len(ub_list) != n:
            raise ValueError("ub must have the same length as constraint")
        if lb_list is not None and len(lb_list) != n:
            raise ValueError("lb must have the same length as constraint")

        eq_terms = []
        ineq_terms = []

        if ub_list is not None and lb_list is not None and all(abs(u - l) < self.tol for u, l in zip(ub_list, lb_list)):
            for i in range(n):
                eq_terms.append(constraint[i] - ub_list[i])
        else:
            if ub_list is not None:
                for i in range(n):
                    ineq_terms.append(constraint[i] - ub_list[i])
            if lb_list is not None:
                for i in range(n):
                    ineq_terms.append(lb_list[i] - constraint[i])

        if eq_terms:
            eq_vec = ca.vertcat(*eq_terms)
            self._terminal_eq_constr.append(eq_vec)
            self._terminal_eq_constr_fn.append(
                ca.Function(f"tec_{len(self._terminal_eq_constr)}", [self.state, self.parameters], [eq_vec])
            )

        if ineq_terms:
            ineq_vec = ca.vertcat(*ineq_terms)
            self._terminal_ineq_constr.append(ineq_vec)
            self._terminal_ineq_constr_fn.append(
                ca.Function(f"tic_{len(self._terminal_ineq_constr)}", [self.state, self.parameters], [ineq_vec])
            )

    def add_terminal_cost(self, cost: SXOrMX) -> None:
        if not isinstance(cost, (ca.MX, ca.SX)):
            raise TypeError("terminal cost must be a CasADi symbolic expression")
        self._terminal_cost = cost
        self._terminal_cost_fn = ca.Function("terminal_cost", [self.state, self.parameters], [self._terminal_cost])
        
    def _stack_variables(self):
        if not self._w:
            return ca.SX.zeros(0, 1)
        self._w = ca.vertcat(*[v.reshape((-1, 1)) for v in self._w])

    def _stack_constraints(self):
        if not self._g:
            return ca.SX.zeros(0, 1)
        self._g = ca.vertcat(*[c.reshape((-1, 1)) for c in self._g])
    
    def _build_problem(self, initial_state: Sequence[float], horizon: int) -> None:
        state_name: str = self.state[0].name() if self.state.numel() > 1 else self.state.name()
        state_sym_type = ca.SX if isinstance(self.state, ca.SX) else ca.MX
        control_name: str = self.control[0].name() if self.control.numel() > 1 else self.control.name()
        control_sym_type = ca.SX if isinstance(self.control, ca.SX) else ca.MX

        self._lbw.clear()
        self._ubw.clear()
        self._lbg.clear()
        self._ubg.clear()
        self._w.clear()
        self._w0.clear()
        self._g.clear()
        self._obj = 0

        x: SXOrMX = state_sym_type.sym(f"{state_name}_0", *self.state.shape)
        self._w.append(x)
        self._w0 += list(initial_state)
        self._lbw += list(initial_state)
        self._ubw += list(initial_state)

        for t in range(horizon):
            u: SXOrMX = control_sym_type.sym(f"{control_name}_{t}", *self.control.shape)
            
            u0: List[float] = [0.5 * (l + u) for l, u in zip(self._control_lb, self._control_ub)]

            self._w.append(u)
            self._w0 += u0
            self._lbw += self._control_lb
            self._ubw += self._control_ub

            if self._stage_eq_constr:
                self._g += [fn(x, u, self.parameters) for fn in self._stage_eq_constr_fn]
                n_eq = sum([c.numel() for c in self._stage_eq_constr])
                self._lbg += [0] * n_eq
                self._ubg += [0] * n_eq

            if self._stage_ineq_constr:
                self._g += [fn(x, u, self.parameters) for fn in self._stage_ineq_constr_fn]
                n_ineq = sum([c.numel() for c in self._stage_ineq_constr])
                self._lbg += [-ca.inf] * n_ineq
                self._ubg += [0] * n_ineq

            if self._stage_cost:
                self._obj += self._stage_cost_fn(x, u, self.parameters)

            x_next: SXOrMX = self.dynamics_fn(x, u, self.parameters)

            x = state_sym_type.sym(f"{state_name}_{t+1}", *self.state.shape)

            self._w.append(x)
            self._w0 += [0] * x.numel()
            self._lbw += self._state_lb
            self._ubw += self._state_ub

            self._g += [x_next - x]
            self._lbg += [0] * x.numel()
            self._ubg += [0] * x.numel()

        if self._terminal_eq_constr:
            self._g += [fn(x, self.parameters) for fn in self._terminal_eq_constr_fn]
            self._lbg += [0] * self._terminal_eq_constr.numel()
            self._ubg += [0] * self._terminal_eq_constr.numel()

        if self._terminal_ineq_constr:
            self._g += [fn(x, self.parameters) for fn in self._terminal_ineq_constr_fn]
            self._lbg += [-ca.inf] * self._terminal_ineq_constr.numel()
            self._ubg += [0] * self._terminal_ineq_constr.numel()

        self._obj += self._terminal_cost_fn(x, self.parameters)

    def _compute_differential(self): 
        trajectory = ca.vertcat(self.state, self.control)
        
        for i, sec in enumerate(self._stage_eq_constr):
            j = ca.jacobian(sec, trajectory)
            self._stage_eq_constr_jac_traj.append(j)
            self._stage_eq_constr_jac_traj_fn.append(ca.Function(f"secjs_{i}" , [self.state, self.control, self.parameters], [j]))
            
            j = ca.jacobian(sec, self.parameters)
            self._stage_eq_constr_jac_params.append(j)
            self._stage_eq_constr_jac_params_fn.append(ca.Function(f"secjp_{i}" , [self.state, self.control, self.parameters], [j]))
            
        for i, sic in enumerate(self._stage_ineq_constr):
            # TODO: I should also consider the state and control bounds here 
            j = ca.jacobian(sic, trajectory)
            self._stage_ineq_constr_jac_traj.append(j)
            self._stage_ineq_constr_jac_traj_fn.append(ca.Function(f"sicjs_{i}" , [self.state, self.control, self.parameters], [j]))
            
            j = ca.jacobian(sic, self.parameters)
            self._stage_ineq_constr_jac_params.append(j)
            self._stage_ineq_constr_jac_params_fn.append(ca.Function(f"sicjp_{i}" , [self.state, self.control, self.parameters], [j]))
            
        for i, tec in enumerate(self._terminal_eq_constr):
            j = ca.jacobian(tec, self.state)
            self._terminal_eq_constr_jac_state.append(j)
            self._terminal_eq_constr_jac_state_fn.append(ca.Function(f"tecjs_{i}" , [self.state, self.parameters], [j]))
            
            j = ca.jacobian(tec, self.parameters)
            self._terminal_eq_constr_jac_params.append(j)
            self._terminal_eq_constr_jac_params_fn.append(ca.Function(f"tecjp_{i}" , [self.state, self.parameters], [j]))
            
        for i, tic in enumerate(self._terminal_ineq_constr):
            # TODO: I should also consider the state and control bounds here 
            j = ca.jacobian(tic, self.state)
            self._terminal_ineq_constr_jac_state.append(j)
            self._terminal_ineq_constr_jac_state_fn.append(ca.Function(f"ticjs_{i}" , [self.state, self.parameters], [j]))
            
            j = ca.jacobian(tic, self.parameters)
            self._terminal_ineq_constr_jac_params.append(j)
            self._terminal_ineq_constr_jac_params_fn.append(ca.Function(f"ticjp_{i}" , [self.state, self.parameters], [j]))
            
        j = ca.jacobian(self.dynamics, trajectory)
        self._dynamics_jac_traj = j
        self._dynamics_jac_traj_fn = ca.Function("djs", [self.state, self.control, self.parameters], [j])
        
        j = ca.jacobian(self.dynamics, self.parameters)
        self._dynamics_jac_params = j
        self._dynamics_jac_params_fn = ca.Function("djp", [self.state, self.control, self.parameters], [j])
                    
        sym_type = ca.SX if isinstance(self.state, ca.SX) else ca.MX
        lambda_d = sym_type.sym("lambda_d", self.state.numel())
        
        lambda_es = [sym_type.sym(f"lambda_e_{i}", seq.numel()) for i, seq in enumerate(self._stage_eq_constr)]
        lambda_is = [sym_type.sym(f"lambda_i_{i}", siq.numel()) for i, siq in enumerate(self._stage_ineq_constr)]

        self._stage_L = ca.dot(lambda_d, self.dynamics)
        if self._stage_cost:
            self._stage_L += self._stage_cost
        for i, seq in enumerate(self._stage_eq_constr):
            self._stage_L += ca.dot(lambda_es[i], seq)
        for i, siq in enumerate(self._stage_ineq_constr):
            self._stage_L += ca.dot(lambda_is[i], siq)

        all_multipliers = lambda_es + lambda_is + [lambda_d]

        self._stage_L_H, stage_L_grad = ca.hessian(self._stage_L, trajectory)
        self._stage_L_H_fn = ca.Function("sLH", [self.state, self.control, self.parameters] + all_multipliers, [self._stage_L_H])
        self._stage_L_H_p = ca.jacobian(stage_L_grad, self.parameters)
        self._stage_L_H_p_fn = ca.Function("sLHp", [self.state, self.control, self.parameters] + all_multipliers, [self._stage_L_H_p])

        if self._terminal_cost is not None:
            sym_type = ca.SX if isinstance(self._terminal_cost, ca.SX) else ca.MX
            
            self._terminal_L = self._terminal_cost

            for i, teq in enumerate(self._terminal_eq_constr):
                lambda_e = sym_type.sym(f"lambda_e_{i}", teq.numel())
                self._terminal_L += ca.dot(lambda_e, teq) 
                
            for i, tiq in enumerate(self._terminal_ineq_constr):
                lambda_i = sym_type.sym(f"lambda_i_{i}", tiq.numel())
                self._terminal_L += ca.dot(lambda_i, tiq) 
                
            self._terminal_L_H, terminal_L_grad = ca.hessian(self._terminal_L, self.state)
            self._terminal_L_H_fn = ca.Function("tLH", [self.state, self.parameters], [self._terminal_L_H])
            self._terminal_L_H_p = ca.jacobian(terminal_L_grad, self.parameters)
            self._terminal_L_H_p_fn = ca.Function("tLHp", [self.state, self.parameters], [self._terminal_L_H_p])
            
    def build(self, initial_state: Sequence[float], horizon: int, solver_name: str = "ipopt", solver_opts: Optional[Dict] = None) -> None:
        assert horizon > 0, "Horizon must be positive"
        
        self._build_problem(initial_state, horizon)
        self._last_horizon = horizon
        
        self._stack_variables()
        self._stack_constraints()
        
        nlp = {"x": self._w, "p": self.parameters, "f": self._obj, "g": self._g}
        opts = {} if solver_opts is None else solver_opts
        self._solver = ca.nlpsol("nlp_solver", solver_name, nlp, opts)
        
        self._compute_differential()

        
    def solve(self, p_vals: Sequence[float]) -> Dict[str, Any]:
        if not self._solver:
            raise RuntimeError("Problem not built. Call `build()` before `solve()`.")
        
        assert len(p_vals) == self.parameters.numel(), (
           f"Parameter size mismatch: expected {self.parameters.numel()} values, got {len(p_vals)}."
        )
        arg = {
            "p": p_vals,
            "x0": self._w0,
            "lbx": self._lbw,
            "ubx": self._ubw,
            "lbg": self._lbg,
            "ubg": self._ubg,
        }
        res = self._solver(**arg)
                
        # Check whether the KKT conditions hold.
        self._converged = self._solver.stats()["return_status"] == CASADI_RETURN_OK

        nx = self.state.numel()
        nu = self.control.numel()

        x_opt = np.array(res["x"].full()).ravel()
        N = (len(x_opt) - nx) // (nx + nu)

        x_padded = np.concatenate([x_opt, np.full(nu, np.nan)])
        x_arr = x_padded.reshape(N + 1, nx + nu)
        x_states = x_arr[:, :nx]
        x_control = x_arr[:-1, nx:]

        lam_g = np.array(res["lam_g"].full()).ravel() if "lam_g" in res else None
        lam_g_stage = None
        lam_g_terminal = None
        
        if lam_g is not None:
            tcn = sum(teq.numel() for teq in self._terminal_eq_constr) + \
                  sum(tiq.numel() for tiq in self._terminal_ineq_constr)
            
            lam_g_terminal = lam_g[-tcn:] if tcn > 0 else np.array([])
            lam_g_stage = lam_g[:-tcn] if tcn > 0 else lam_g
            lam_g_stage = lam_g_stage.reshape(N, -1)

        lam_x = np.array(res["lam_x"].full()).ravel() if "lam_x" in res else None

        self._last_solution = {"x_raw": x_opt,
                               "state": x_states,
                               "control": x_control,
                               "lam_g": lam_g,
                               "lam_g_stage": lam_g_stage,
                               "lam_g_terminal": lam_g_terminal,
                               "lam_x": lam_x,
                               "p": p_vals,
                               "obj": res["f"]}

        return deepcopy(self._last_solution)

    def _group_lambdas(self, lambda_t: np.ndarray, is_stage: bool) -> List[float]:
        lambda_t_grouped = []
        idx = 0
        
        eq_cstr = self._stage_eq_constr if is_stage else self._terminal_eq_constr
        ineq_cstr = self._stage_ineq_constr if is_stage else self._terminal_ineq_constr
        
        for eq in eq_cstr:
            lambda_t_grouped.append(lambda_t[idx : idx+eq.numel()])
            idx += eq.numel()
            
        for ineq in ineq_cstr:
            lambda_t_grouped.append(lambda_t[idx : idx+ineq.numel()])
            idx += ineq.numel()
        
        if is_stage:
            lambda_t_grouped.append(lambda_t[idx : idx+self.dynamics.numel()])
            idx += self.dynamics.numel()
        
        assert idx == lambda_t.size, "[DEBUG] Something went wrong..."
        
        return lambda_t_grouped

    def _build_block_idoc(self):
        p = self._last_solution["p"]
        At = []
        Bt = []
        Ct = []
        Ht = []
        
        for t in range(self._last_horizon):
            xt = self._last_solution["state"][t]
            ut = self._last_solution["control"][t]
            lambda_t = self._last_solution["lam_g_stage"][t]
            lambda_t_grouped = self._group_lambdas(lambda_t, is_stage=True)
            
            Bt.append(self._stage_L_H_p_fn(xt, ut, p, *lambda_t_grouped))
            Ht.append(self._stage_L_H_fn(xt, ut, p, *lambda_t_grouped))
            
            r_eps_t = []
            r_p_t = []
            
            if self._stage_eq_constr:
                r_eps_t += [f(xt, ut, p) for f in self._stage_eq_constr_jac_traj_fn]
                r_p_t += [f(xt, ut, p) for f in self._stage_eq_constr_jac_params_fn]
            
            if self._stage_ineq_constr:
                ne = len(self._stage_eq_constr)
                ni = len(self._stage_ineq_constr)
                lambda_t_ineq = lambda_t_grouped[ne : ne + ni]
                for i in range(ni):
                    ft = self._stage_ineq_constr_jac_traj_fn[i]
                    fp = self._stage_ineq_constr_jac_params_fn[i]
                    lambda_t_i = lambda_t_ineq[i]
                    mask = np.abs(lambda_t_i) < self.tol
                    r_eps_t += ft(xt, ut, p)[mask]
                    r_p_t += fp(xt, ut, p)[mask]
                    
            r_eps_t.append(self._dynamics_jac_traj_fn(xt, ut, p))
            r_p_t.append(self._dynamics_jac_params_fn(xt, ut, p))
            
            At.append(np.vstack(r_eps_t) if r_eps_t else np.array([]))
            Ct.append(np.vstack(r_p_t) if r_p_t else np.array([]))
            
        xT = self._last_solution["state"][-1]
        lambda_T = self._last_solution["lam_g_terminal"]
        lambda_T_grouped = self._group_lambdas(lambda_T, is_stage=False)
        
        BT = self._terminal_L_H_p_fn(xT, p, *lambda_T_grouped)
        HT = self._terminal_L_H_fn(xT, p, *lambda_T_grouped)
        
        r_eps_T = []
        r_p_T = []
        
        if self._terminal_eq_constr:
            r_eps_T += [f(xT, p) for f in self._terminal_eq_constr_jac_state_fn]
            r_p_T += [f(xT, p) for f in self._terminal_eq_constr_jac_params_fn]
            
        if self._terminal_ineq_constr:
            ne = len(self._terminal_eq_constr)
            ni = len(self._terminal_ineq_constr)
            lambda_T_ineq = lambda_T_grouped[ne : ne + ni]
            for i in range(ni):
                ft = self._terminal_ineq_constr_jac_state_fn[i]
                fp = self._terminal_ineq_constr_jac_params_fn[i]
                lambda_T_i = lambda_T_ineq[i]
                mask = np.abs(lambda_T_i) < self.tol
                r_eps_T += ft(xT, p)[mask]
                r_p_T += fp(xT, p)[mask]
        
        AT = np.vstack(r_eps_T) if r_eps_T else np.array([])
        CT = np.vstack(r_p_T) if r_p_T else np.array([])
        
        return At, AT, Bt, BT, Ct, CT, Ht, HT

    def backward(self, inversion_method: str = "pseudo") -> np.ndarray:
        if not self._converged:
            raise RuntimeError("Cannot compute sensitivity: the last solve did not converge to a KKT-satisfying point.")

        inversion_methods = {
            "exact": np.linalg.inv,
            "pseudo": np.linalg.pinv,
            "regularization": lambda x: np.linalg.inv(x + np.eye(x.shape[0]) * 1e-3),
        }

        inv = inversion_methods.get(inversion_method)
        if inv is None:
            raise ValueError(f"Unknown inversion method: {inversion_method}")

        At, AT, Bt, BT, Ct, CT, Ht, HT = self._build_block_idoc()
        
        Hinv_t = inv(Ht)
        Hinv_T = inv(HT)
        

ocp = GradOCP()
dt = 1
x,u,y = ca.SX.sym("x",2), ca.SX.sym("u",2), ca.SX.sym("y",2)
ocp.add_state(x)
ocp.add_control(u, [0,0], [2*np.pi, 2*np.pi])
ocp.add_parameters(y)

x_next = x + dt*(x + u * y * ca.cos(u))

ocp.add_dynamics(x_next)

ocp.add_terminal_cost(x[0]**2 + x[1]**2)

ocp.build([0,0], 3, solver_opts={"ipopt.print_level":0, "print_time":0, "ipopt.sb":"yes"})
res = ocp.solve([100,100])

#pprint.pprint(res, width = 20)
ocp.backward()