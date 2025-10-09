from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import casadi as ca

@dataclass
class DecisionVariable:
    name: str
    size: int
    stage: int 
    lb: Optional[List[float]] = None
    ub: Optional[List[float]] = None
    symbol: ca.SX = field(init=False)

    def __post_init__(self):
        self.symbol = ca.SX.sym(self.name, self.size)
        if self.lb is None:
            self.lb = [-ca.inf] * self.size
        if self.ub is None:
            self.ub = [ ca.inf] * self.size
        if any(l > u for l, u in zip(self.lb, self.ub)):
            raise ValueError(
                f"DecisionVariable '{self.name}' has lower bound(s) {self.lb} exceeding upper bound(s) {self.ub}"
            )
        if any(abs(u - l) < 1e-8 for l, u in zip(self.lb, self.ub)):
            raise ValueError(
                f"DecisionVariable '{self.name}' with size {self.size} has some bounds that are effectively equal "
                f"(lb={self.lb}, ub={self.ub}). Use a Parameter or a constant instead."
            )

    def __add__(self, other):
        return self.symbol + other
    def __radd__(self, other):
        return other + self.symbol
    def __sub__(self, other):
        return self.symbol - other
    def __rsub__(self, other):
        return other - self.symbol
    def __mul__(self, other):
        return self.symbol * other
    def __rmul__(self, other):
        return other * self.symbol
    def __truediv__(self, other):
        return self.symbol / other
    def __rtruediv__(self, other):
        return other / self.symbol
    def __getitem__(self, idx):
        return self.symbol[idx]

@dataclass
class Parameter:
    name: str
    size: int = 1
    symbol: Any = field(init=False)

    def __post_init__(self):
        self.symbol = ca.SX.sym(self.name, self.size)

    def __add__(self, other):
        return self.symbol + other
    def __radd__(self, other):
        return other + self.symbol
    def __sub__(self, other):
        return self.symbol - other
    def __rsub__(self, other):
        return other - self.symbol
    def __mul__(self, other):
        return self.symbol * other
    def __rmul__(self, other):
        return other * self.symbol
    def __truediv__(self, other):
        return self.symbol / other
    def __rtruediv__(self, other):
        return other / self.symbol
    def __getitem__(self, idx):
        return self.symbol[idx]

@dataclass
class Constraint:
    # TODO: support vectors
    expr: ca.SX
    stage: int
    lb: float = 0.0
    ub: float = 0.0

    def __post_init__(self):
        if self.lb > self.ub:
            raise ValueError(f"Constraint lower bound {self.lb} exceeds upper bound {self.ub}")

    def is_equality(self, tol = 1e-8):
        return self.ub-self.lb < tol
