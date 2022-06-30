# import gurobipy as gp
# from gurobipy import GRB
import numpy as np
from cvxopt import matrix, solvers
from cvxopt.modeling import dot


def optimize_star(c, A, b, mode='min', solver='cvx-glpk'):
    if solver == 'gurobi':
        raise ValueError("activate after enabling gurobi license")
        # return optimize_star_gurobi(c, A, b, mode)
    elif solver == 'cvx-glpk':
        return optimize_star_cvx_glpk(c, A, b, mode)
    else:
        raise ValueError("Unrecognized solver name used")


def isFeasible_star(A, b, solver='cvx-glpk'):
    if solver == 'gurobi':
        raise ValueError("activate after enabling gurobi license")
        # return isFeasible_star_gurobi(A, b)
    elif solver == 'cvx-glpk':
        return isFeasible_star_cvx_glpk(A, b)
    else:
        raise ValueError("Unrecognized solver name used")


def optimize_star_cvx_glpk(c, A, b, mode='min'):
    if mode == 'min':
        c = matrix(c)
    elif mode == 'max':
        c = matrix(-1.0 * c)
    else:
        print('Invalid Mode')

    A = matrix(A)
    b = matrix(b)

    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    sol = solvers.lp(c, A, b, solver='glpk')

    optimum = sol['x']

    if optimum[0] is None:
        print('Infeasible or unbounded solution')
    else:
        if mode == 'min':
            obj_val = dot(optimum, c)
            return obj_val
        elif mode == 'max':
            obj_val = dot(optimum, -1.0 * c)
            return obj_val
        else:
            print('Invalid mode')


'''
def optimize_star_gurobi(c, A, b, mode='min'):
    num_var = A.shape[1]
    m = gp.Model()
    m.Params.OutputFlag = 0
    variables = m.addMVar(vtype=GRB.CONTINUOUS, shape=num_var, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    m.addConstr(A @ variables <= b)

    if mode == 'min':
        m.setObjective(c @ variables, GRB.MINIMIZE)
    elif mode == 'max':
        m.setObjective(c @ variables, GRB.MAXIMIZE)

    m.optimize()

    if m.Status == 2:
        return m.objVal
    else:
        print('Errors, model not optimally solved. Status code is ' + str(m.Status))

'''


def isFeasible_star_cvx_glpk(A, b):
    num_var = A.shape[1]
    dummy_obj = matrix(1.0 * np.zeros(num_var))
    A = matrix(A)
    b = matrix(b)

    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    sol = solvers.lp(dummy_obj, A, b, solver='glpk')

    optimum = sol['x']

    if optimum is None:
        return False
    else:
        return True


'''
def isFeasible_star_gurobi(A, b):
    num_var = A.shape[1]
    m = gp.Model()
    m.Params.OutputFlag = 0
    variables = m.addMVar(vtype=GRB.CONTINUOUS, shape=num_var, lb=-1 * GRB.INFINITY, ub=GRB.INFINITY)

    dummy_obj = np.zeros(num_var)

    m.addConstr(A @ variables <= b)
    m.setObjective(dummy_obj @ variables, GRB.MINIMIZE)

    m.optimize()

    if m.Status == 3:
        return False
    elif m.Status == 2:
        return True
    else:
        print('Error, model not optimally solved. Status code is ' + str(m.Status))
        return False



c = np.array([-4.0, -5.0])
A = np.array([[2.0, 1.0], [1.0, 2.0], [-1.0, 0.0], [0.0, -1.0], [-2.0, -1.0]])
b = np.array([3.0, 3.0, 0.0, 0.0, -4.0])
print(isFeasible_star(A, b))
'''
