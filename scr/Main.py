import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from ORLibaryDataSet import ORLibaryDataSet
from LagrangianRelaxation import LagrangianRelaxation
from LagrangianRelaxation import SurrogateLagrangianRelaxation
from Heuristic import HeuristicBySolver

data_set = ORLibaryDataSet('GenAssignProblem')
n_agent, n_job, A, capacity, cost_coeff = data_set.gen_assign_prob(file_name = 'b05200.txt')

model = gp.Model('GenAssignProblem')
x = model.addVars(n_agent, n_job, name = 'x', vtype = GRB.BINARY)

relaxed_expr = []
for i in range(n_agent):
    relaxed_expr.append(gp.LinExpr(gp.quicksum(A[i,j] * x[i,j] for j in range(n_job)) - capacity[i])) 

constrs1 = model.addConstrs(gp.quicksum(A[i,j] * x[i,j] for j in range(n_job)) <= capacity[i] for i in range(n_agent))
constrs2 = model.addConstrs(gp.quicksum(x[i,j] for i in range(n_agent)) == 1 for j in range(n_job))

obj = gp.LinExpr()
for i in range(n_agent):
    obj += gp.quicksum(cost_coeff[i,j] * x[i,j] for j in range(n_job))

model.setObjective(obj, GRB.MAXIMIZE)
model.update()
model.write('model.lp')

# input original model, relaxed constrs and initialization muliper
relaxed_constrs, n_relaxed = [constrs1], n_agent
mulpier = -0.02 * np.zeros((1, n_relaxed))
heuristic_solver = HeuristicBySolver(model, gap = 0.01, time_limit = 300)

# SurrogateLagrangianRelaxation 表示代理拉格朗日松弛法, LagrangianRelaxation 表示普通拉格朗日松弛法
lagrangian_relaxation = LagrangianRelaxation(model, relaxed_constrs, mulpier)
lagrangian_relaxation.bulid_relaxed_duality(lagrangian_relaxation)

#lagrangian_relaxation = SurrogateLagrangianRelaxation(model, relaxed_constrs, mulpier, r = 0.8, big_m = 1.2)
#lagrangian_relaxation.bulid_relaxed_duality(lagrangian_relaxation)

max_iter_times = 30

for k in range(max_iter_times):
    lagrangian_relaxation.relaxed_prob.reset_relaxed_objective(mulpier)   # 更新松弛问题目标函数
    lagrangian_relaxation.relaxed_prob.optimize()                         # 求解松弛问题
    relaxed_expr = lagrangian_relaxation.relaxed_prob.get_relaxed_expr()    # 输出松弛约束表达式
    relaxed_obj_values = lagrangian_relaxation.relaxed_prob.get_objective_values()  # 输出松弛问题的目标函数
    
    lagrangian_relaxation.relaxed_prob.write_model(k)
    if lagrangian_relaxation.is_feasible():
        break

    subgradients = lagrangian_relaxation.duality_prob.get_subgradients(relaxed_expr)  # 计算次梯度
    lagrangian_relaxation.duality_prob.get_step_size(k, 0.8, relaxed_obj_values, heuristic_solver)   # 计算次梯度步长
    mulpier = lagrangian_relaxation.duality_prob.update_mulpier()                     # 更新乘子

    lagrangian_relaxation.print_status(k)  
    lagrangian_relaxation.save_kpi()

lagrangian_relaxation.plot_kpi()