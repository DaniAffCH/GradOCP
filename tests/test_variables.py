import pytest
import casadi as ca
from GradOCP.variables import DecisionVariable, Parameter, Constraint

def test_decision_variable_creation():
    x = DecisionVariable("x", size=2, stage=0, lb=[0,0], ub=[1,2])
    assert x.symbol.size1() == 2
    assert x.lb == [0,0]
    assert x.ub == [1,2]

def test_decision_variable_invalid_bounds():
    with pytest.raises(ValueError):
        DecisionVariable("x", stage=0, size=1, lb=[2], ub=[1])

def test_parameter_creation():
    p = Parameter("p", size=3)
    assert p.symbol.size1() == 3

def test_constraint_equality_detection():
    c = Constraint(ca.SX.sym("x"), stage=0, lb=1.0, ub=1.0)
    assert c.is_equality()
