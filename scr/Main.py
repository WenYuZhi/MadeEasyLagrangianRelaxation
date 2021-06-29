import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from ORLibaryDataSet import ORLibaryDataSet
from LagrangianRelaxation import LagrangianRelaxation

data_set = ORLibaryDataSet('GenAssignProblem')
n_agent, n_job, A, capacity, cost_coeff = data_set.gen_assign_prob(file_name = 'a05100.txt')

m = gp.Model('GAP')
x = m.addVars(n_agent, n_job, name = 'x', vtype = GRB.BINARY)

relaxed_expr = []
for i in range(n_agent):
    relaxed_expr.append(gp.LinExpr(gp.quicksum(A[i,j] * x[i,j] for j in range(n_job)) - capacity[i])) 

constrs1 = m.addConstrs(gp.quicksum(A[i,j] * x[i,j] for j in range(n_job)) <= capacity[i] for i in range(n_agent))
constrs2 = m.addConstrs(gp.quicksum(x[i,j] for i in range(n_agent)) == 1 for j in range(n_job))

obj = gp.LinExpr()
for i in range(n_agent):
    obj += gp.quicksum(cost_coeff[i,j] * x[i,j] for j in range(n_job))

m.setObjective(obj, GRB.MINIMIZE)
m.optimize()
m.write('model.lp')

relaxed_constrs, n_relaxed = [constrs2], n_job
mulpier = np.zeros((1, n_relaxed))
lagrangian_relaxation = LagrangianRelaxation(m, relaxed_constrs, n_relaxed, mulpier)

for k in range(5):
    lagrangian_relaxation.step(k)
