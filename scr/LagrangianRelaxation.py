import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import matplotlib.pyplot as plt

class LagrangianRelaxation:
    def __init__(self, model, relaxed_constrs, mulpier):
        self.orig_model, self.mulpier = model.copy(), mulpier
        self.relaxed_ind, self.n_relaxed = self.__get_relaxed_constrs(relaxed_constrs)
        self.relaxed_objective_values_trace, self.gap_trace, self.step_size_trace = [], [], []
        assert(mulpier.size == self.n_relaxed)
        assert(len(self.relaxed_ind) == self.n_relaxed)
    
    def bulid_relaxed_duality(self, lagrangian_relaxation):
        self.relaxed_prob = self.RelaxedProb(lagrangian_relaxation)
        self.relaxed_expr = self.relaxed_prob.get_relaxed_expr()
        self.sense = self.relaxed_prob.get_sense()
        self.duality_prob = self.DualityProb(lagrangian_relaxation)
        
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
    
    def is_feasible(self):
        self.relaxed_expr_value = np.array([x.getValue() for x in self.relaxed_prob.get_relaxed_expr()])
        if (self.relaxed_expr_value[self.relaxed_prob.get_sense() == '<'] <= 0).all() and \
        (self.relaxed_expr_value[self.relaxed_prob.get_sense() == '>'] >= 0).all() and \
        (self.relaxed_expr_value[self.relaxed_prob.get_sense() == '='] == 0).all():
            print("relaxed problem has obtained feasible solution")
            print(self.relaxed_prob.objective_values)
            return True
        else:
            return False
    
    def get_gap(self):
        self.gap = 100 * self.duality_prob.sign_flag * (self.duality_prob.best_bound - self.relaxed_prob.get_objective_values()) / self.duality_prob.best_bound
        return self.gap
    
    def print_status(self, k):
        print("iteration time: ", k)
        print("subgradients: ", self.duality_prob.subgradients)
        print("mulpier: ", self.duality_prob.mulpier)
        print("relaxed obj values: ", self.relaxed_prob.objective_values)
        print("MIP Gap: " + "{:.5f}".format(self.get_gap()) + "%")
        print("Best Bound:", "{:.5f}".format(self.duality_prob.best_bound))
        print(self.relaxed_expr_value)
        print(" ")
    
    def save_kpi(self):
        self.relaxed_objective_values_trace.append(self.relaxed_prob.objective_values)
        self.gap_trace.append(self.get_gap())
        self.step_size_trace.append(self.duality_prob.step_size)
    
    def plot_kpi(self):
        plt.figure()
        plt.plot(self.relaxed_objective_values_trace)
        plt.ylabel('Lower Bound')
        plt.xlabel('Iteration times')

        plt.figure()
        plt.plot(self.gap_trace)
        plt.ylabel('GAP(%)')
        plt.xlabel('Iteration times')

        plt.figure()
        plt.plot(self.step_size_trace)
        plt.ylabel('Step size(%)')
        plt.xlabel('Iteration times')
        plt.show()


    class RelaxedProb:
        def __init__(self, lagrangian_relaxation):
            self.model, self.relaxed_ind, self.n_relaxed = lagrangian_relaxation.orig_model.copy(), lagrangian_relaxation.relaxed_ind, lagrangian_relaxation.n_relaxed
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
            sense = np.array([self.model.getConstrs()[constrs_ind].getAttr('Sense') for constrs_ind in self.relaxed_ind])
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
    
    class DualityProb:
        def __init__(self, lagrangian_relaxation):
            self.mulpier, self.sense = lagrangian_relaxation.mulpier, lagrangian_relaxation.sense
            self.orig_model_sense = lagrangian_relaxation.orig_model.getAttr("ModelSense")
            self.sign_flag = self.step_size = 1.0 if self.orig_model_sense == 'Minimization' else -1.0
        
        def get_subgradients(self, relaxed_expr):
            self.subgradients = np.array([relaxed_expr[i].getValue() for i in range(len(relaxed_expr))])
            return self.subgradients   
        
        def get_step_size(self, k, relaxed_obj_values, heuristic_solver):
            if k == 0:
                self.best_bound = self.__estimate_best_bound(heuristic_solver)

            self.relaxed_obj_values = relaxed_obj_values
            self.step_size = self.sign_flag * (self.best_bound - relaxed_obj_values) / (np.linalg.norm(self.subgradients))**2

        def update_mulpier(self):
            self.mulpier += self.sign_flag * self.step_size * self.subgradients
            for i in range(self.mulpier.size):
                if self.sense[i] == '<':
                    self.mulpier[0, i] = max(0, self.mulpier[0, i]) if self.sign_flag == 1.0 else min(0, self.mulpier[0, i])
                elif self.sense[i] == '>':
                    self.mulpier[0, i] = min(0, self.mulpier[0, i]) if self.sign_flag == 1.0 else max(0, self.mulpier[0, i])
            return self.mulpier

        def __estimate_best_bound(self, heuristic_solver):
            heuristic_solver.optimize()
            best_bound = heuristic_solver.get_objective_values()
            return best_bound

class SurrogateLagrangianRelaxation(LagrangianRelaxation):
    def __init__(self, model, relaxed_constrs, mulpier, r, big_m):
        super(SurrogateLagrangianRelaxation, self).__init__(model, relaxed_constrs, mulpier)
        self.LagrangianRelaxation = LagrangianRelaxation
        self.r, self.big_m = r, big_m
        assert(self.r > 0 and self.r < 1)
        assert(self.big_m >= 1)
    
    class DualityProb:
        def __init__(self, lagrangian_relaxation):
            self.mulpier, self.sense = lagrangian_relaxation.mulpier, lagrangian_relaxation.sense
            self.orig_model_sense = lagrangian_relaxation.orig_model.getAttr("ModelSense")
            self.sign_flag = self.step_size = 1.0 if self.orig_model_sense == 'Minimization' else -1.0
        
            self.r, self.big_m = lagrangian_relaxation.r, lagrangian_relaxation.big_m
        
        def get_subgradients(self, relaxed_expr):
            self.subgradients = np.array([relaxed_expr[i].getValue() for i in range(len(relaxed_expr))])
            return self.subgradients   
        
        def get_step_size(self, k, relaxed_obj_values, heuristic_solver):
            if k == 0:
                self.best_bound = self.__estimate_best_bound(heuristic_solver)
                self.relaxed_obj_values = relaxed_obj_values
                self.step_size = self.sign_flag * (self.best_bound - self.relaxed_obj_values) / (np.linalg.norm(self.subgradients))**2
            
            else:
                self.p = 1 - 1 / ((k + 1)**self.r)
                self.alpha = 1 - 1 / (self.big_m * (k + 1)**self.p)
                self.step_size = self.alpha * self.last_step_size * np.linalg.norm(self.last_subgradients) / np.linalg.norm(self.subgradients)
            
            self.last_step_size, self.last_subgradients = self.step_size, self.subgradients.copy()
           
        def update_mulpier(self):
            self.mulpier += self.sign_flag * self.step_size * self.subgradients
            for i in range(self.mulpier.size):
                if self.sense[i] == '<':
                    self.mulpier[0, i] = max(0, self.mulpier[0, i]) if self.sign_flag == 1.0 else min(0, self.mulpier[0, i])
                elif self.sense[i] == '>':
                    self.mulpier[0, i] = min(0, self.mulpier[0, i]) if self.sign_flag == 1.0 else max(0, self.mulpier[0, i])
            return self.mulpier

        def __estimate_best_bound(self, heuristic_solver):
            heuristic_solver.optimize()
            best_bound = heuristic_solver.get_objective_values()
            return best_bound
  
        

            
    
    
        