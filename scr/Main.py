import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from ORLibaryDataSet import ORLibaryDataSet
from LagrangianRelaxation import LagrangRelax
from LagrangianRelaxation import SurrogateLagrangRelax
from LagrangianRelaxation import LevLagrangRelax
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

# SurrogateLagrangRelax 表示代理拉格朗日松弛法, LagrangRelax 表示普通拉格朗日松弛法, LevLagrangRelax 表示水平拉格朗日松弛法

#lr = LagrangRelax(model, relaxed_constrs, mulpier)
#lr.bulid_relaxed_duality(lr)
'''
lr = SurrogateLagrangRelax(model, relaxed_constrs, mulpier, r = 0.05, big_m = 30)
lr.bulid_relaxed_duality(lr)

max_iter_times = 2

for k in range(max_iter_times):
    lr.relaxed_prob.reset_relaxed_objective(mulpier)   # 更新松弛问题目标函数
    lr.relaxed_prob.optimize()                         # 求解松弛问题
    relaxed_expr = lr.relaxed_prob.get_relaxed_expr()    # 输出松弛约束表达式
    relaxed_obj_values = lr.relaxed_prob.get_objective_values()  # 输出松弛问题的目标函数
    
    lr.relaxed_prob.write_model(k)
    if lr.is_feasible():
        break

    subgrad = lr.duality_prob.get_subgrad(relaxed_expr)  # 计算次梯度
    lr.duality_prob.get_step_size(k, relaxed_obj_values, heuristic_solver)   # 计算次梯度步长
    mulpier = lr.duality_prob.update_mulpier()                     # 更新乘子

    lr.print_status(k)  
    lr.save_kpi()
'''

lr = LevLagrangRelax(model, relaxed_constrs, mulpier, R = 10**4)
lr.bulid_relaxed_duality(lr)

max_iter_times = 10

for k in range(max_iter_times):
    lr.relaxed_prob.reset_relaxed_objective(mulpier)   # 更新松弛问题目标函数
    lr.relaxed_prob.optimize()                         # 求解松弛问题
    relaxed_expr = lr.relaxed_prob.get_relaxed_expr()    # 输出松弛约束表达式
    relaxed_obj_values = lr.relaxed_prob.get_objective_values()  # 输出松弛问题的目标函数
    
    lr.relaxed_prob.write_model(k)
    if lr.is_feasible():
        break

    subgrad = lr.duality_prob.get_subgrad(relaxed_expr)  # 计算次梯度
    lr.duality_prob.evaluate_objective(relaxed_obj_values, k)
    lr.duality_prob.is_descent(k)
    lr.duality_prob.get_step_size()
    mulpier = lr.duality_prob.update_mulpier()
    lr.duality_prob.update_path()
