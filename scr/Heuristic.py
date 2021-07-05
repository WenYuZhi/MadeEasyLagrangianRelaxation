import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

class HeuristicBySolver:
    def __init__(self, model, gap, time_limit):
            self.model = model.copy()
            self.gap, self.time_limit = gap, time_limit
            self.__set_solver_params()  
        
    def __set_solver_params(self):
        self.model.setParam('MIPGap', self.gap)
        self.model.setParam('TimeLimit', self.time_limit)
        self.model.setParam('OutputFlag', False)
        
    def optimize(self):
        self.model.optimize()

    def get_objective_values(self):
        self.objective_values = self.model.getObjective().getValue()
        return self.objective_values

class HeuristicLPRelaxation:
    def __init__(self, model):
        self.model = model.copy()
        self.model_lp_relaxation = self.model.relax()

    def optimize(self):
        self.model.optimize()
        
    def get_objective_values(self):
        self.objective_values = self.model_lp_relaxation.getObjective().getValue()
        return self.objective_values
        
    def get_solution(self):
        self.solution = [x.x for x in self.model.getVars()]
        return np.array(self.solution)
        
    def local_search(self):
        pass