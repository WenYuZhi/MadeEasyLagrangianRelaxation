import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os

class LagrangianRelaxation:
    def __init__(self, model, relaxed_constrs, n_relaxed, mulpier):
        assert(mulpier.size == n_relaxed)
        self.orig_model, self.mulpier = model.copy(), mulpier
        self.relaxed_ind, self.n_relaxed = self.__get_relaxed_constrs(relaxed_constrs)
        self.relaxed_prob = self.RelaxedProb(model, self.relaxed_ind, self.n_relaxed)
        relaxed_expr = self.relaxed_prob.get_relaxed_expr()
        sense = self.relaxed_prob.get_sense()
        self.dual_prob = self.DualProb(mulpier, sense)

    def __get_relaxed_constrs(self, relaxed_constrs):
        relaxed_constrs_ind = []
        for constr_item in relaxed_constrs:
            if type(constr_item) == gp.gurobipy.tupledict:
                for index in constr_item:
                    relaxed_constrs_ind.append(constr_item[index].index)
            elif type(constr_item) == gp.gurobipy.Constr:
                relaxed_constrs_ind.append(constr_item.index)
            else:
                print("input constraints type error")
        return relaxed_constrs_ind, len(relaxed_constrs_ind)
    
    def step(self, k):
        self.relaxed_prob.reset_relaxed_objective(self.mulpier)
        self.relaxed_prob.optimize()
        self.x = self.relaxed_prob.get_solution()
        self.relaxed_expr = self.relaxed_prob.get_relaxed_expr()
        self.relaxed_obj_values = self.relaxed_prob.get_objective_values()
        self.relaxed_prob.write_model(k)

        self.subgradients = self.dual_prob.get_subgradients(self.relaxed_expr)
        self.dual_prob.get_step_size(self.relaxed_obj_values)
        self.mulpier = self.dual_prob.update_mulpier()
        self.print_status()
    
    def print_status(self):
        print("subgradients: ", self.subgradients)
        print("mulpier: ", self.mulpier)
        print("relaxed obj values: ", self.relaxed_obj_values)


    class RelaxedProb:
        def __init__(self, model, relaxed_ind, n_relaxed):
            assert(len(relaxed_ind) == n_relaxed)
            self.model, self.relaxed_ind, self.n_relaxed = model.copy(), relaxed_ind, n_relaxed
            self.constrs_matrix, self.rhs, self.sense = self.__get_constrs_matrix(), self.__get_constrs_rhs(), self.__get_constrs_sense()
            self.relaxed_expr = self.__get_relaxed_expr()
            self.__remove_relaxed_constrs()
             
        def __get_constrs_matrix(self):
            constrs_matrix = self.model.getA()[self.relaxed_ind, :]
            return constrs_matrix.todense()
        
        def __get_constrs_rhs(self):
            rhs = [self.model.getConstrs()[constrs_ind].getAttr('RHS') for constrs_ind in self.relaxed_ind]
            return np.array(rhs)
        
        def __get_constrs_sense(self):
            sense = [self.model.getConstrs()[constrs_ind].getAttr('Sense') for constrs_ind in self.relaxed_ind]
            return sense
        
        def __get_relaxed_expr(self):
            relaxed_expr = [gp.LinExpr() for i in range(self.n_relaxed)]
            for i in range(self.n_relaxed):
                for j in range(self.constrs_matrix.shape[1]):
                    relaxed_expr[i].addTerms(self.constrs_matrix[i,j], self.model.getVars()[j])
                relaxed_expr[i].addConstant(-1 * self.rhs[i])
            return relaxed_expr
        
        def get_relaxed_expr(self):
            return self.relaxed_expr
        
        def get_sense(self):
            return self.sense

        def __remove_relaxed_constrs(self):
            for constrs_ind in self.relaxed_ind:
                self.model.remove(self.model.getConstrs()[constrs_ind])
            self.model.update()
        
        def __get_relaxed_objective(self, mulpier):
            objective = self.model.getObjective()
            for i in range(self.n_relaxed):
                objective.add(self.relaxed_expr[i], mulpier[0,i])    
            return objective 
            
        def __optimize_by_gurobi(self):
            self.model.Params.OutputFlag = 0
            self.model.optimize()
        
        def optimize(self, method = 'Gurobi'):
            if method == 'Gurobi':
                self.__optimize_by_gurobi()
        
        def reset_relaxed_objective(self, mulpier):
            objective = self.__get_relaxed_objective(mulpier)
            self.model.setObjective(objective)

        def get_solution(self):
            self.solution = [x.x for x in self.model.getVars()]
            return np.array(self.solution)
        
        def get_objective_values(self):
            self.objective_values = self.model.getObjective().getValue()
            return self.objective_values
        
        def write_model(self, k):
            self.route = os.getcwd() + '\\results\\'
            self.model.write(self.route + 'GAPModel' + str(k) + '.lp')
    
    class DualProb:
        def __init__(self, mulpier, sense):
            self.mulpier, self.sense = mulpier, sense
            self.best_ub = self.__estimate_best_ub()
        
        def get_subgradients(self, relaxed_expr):
            self.subgradients = np.array([relaxed_expr[i].getValue() for i in range(len(relaxed_expr))])
            return self.subgradients   
        
        def get_step_size(self, relaxed_obj_values):
            self.relaxed_obj_values = relaxed_obj_values
            self.step_size = (self.best_ub - self.relaxed_obj_values) / (np.linalg.norm(self.subgradients))**2

        def update_mulpier(self):
            self.mulpier += self.step_size * self.subgradients
            for i in range(self.mulpier.size):
                if self.sense[i] == '<':
                    self.mulpier[0, i] = min(0, self.mulpier[i])
                elif self.sense[i] == '>':
                    self.mulpier[0, i] = max(0, self.mulpier[i])
            return self.mulpier

        def __estimate_best_ub(self):
            best_ub = 1700
            return best_ub

        #if iterTimes == 0:
            #self.stepSize = (self.bestUpBound - self.relaxedObjValues) / (np.linalg.norm(self.subgradients))**2
        #else:
            #self.p = 1 - 1 / ((iterTimes + 1)**self.r)
            #self.alpha = 1 - 1 / (self.m * (iterTimes + 1)**self.p)
            #self.stepSize = self.alpha * self.stepSizeLast * np.linalg.norm(self.subgrdientsLast) / np.linalg.norm(self.subgradients)
    

        