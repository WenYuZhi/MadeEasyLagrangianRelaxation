import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import matplotlib.pyplot as plt

class LagrangRelax:
    def __init__(self, model, relaxed_constrs, mulpier):
        self.orig_model, self.mulpier = model.copy(), mulpier
        self.relaxed_ind, self.n_relaxed = self.__get_relaxed_constrs(relaxed_constrs)
        self.relaxed_objective_values_trace, self.gap_trace, self.step_size_trace = [], [], []
        assert(mulpier.size == self.n_relaxed)
        assert(len(self.relaxed_ind) == self.n_relaxed)
    
    def bulid_relaxed_duality(self, lagrang_relax):
        self.relaxed_prob = self.RelaxedProb(lagrang_relax)
        self.relaxed_expr = self.relaxed_prob.get_relaxed_expr()
        self.sense = self.relaxed_prob.get_sense()
        self.duality_prob = self.DualityProb(lagrang_relax)
        
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
        self.gap = 100 * self.duality_prob.sign * (self.duality_prob.best_bound - self.relaxed_prob.get_objective_values()) / self.duality_prob.best_bound
        return self.gap
    
    def print_status(self, k):
        print("iteration time: ", k)
        print("subgrad: ", self.duality_prob.subgrad)
        print("mulpier: ", self.duality_prob.mulpier)
        print("relaxed obj values: ", self.relaxed_prob.objective_values)
        print("MIP Gap: " + "{:.5f}".format(self.get_gap()) + "%")
        print("best bound: " + "{:.5f}".format(self.duality_prob.best_bound))
        print("step size: " + "{:.5f}".format(self.duality_prob.step_size))
        print("relaxed expr value: ", self.relaxed_expr_value)
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
        def __init__(self, lagrang_relax):
            self.model, self.relaxed_ind, self.n_relaxed = lagrang_relax.orig_model.copy(), lagrang_relax.relaxed_ind, lagrang_relax.n_relaxed
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
        def __init__(self, lagrang_relax):
            self.mulpier, self.sense = lagrang_relax.mulpier.copy(), lagrang_relax.sense
            self.sign = 1.0 if lagrang_relax.orig_model.getAttr("ModelSense") == 'Minimization' else -1.0
        
        def get_subgrad(self, relaxed_expr):
            self.subgrad = np.array([relaxed_expr[i].getValue() for i in range(len(relaxed_expr))])
            return self.subgrad   
        
        def get_step_size(self, k, relaxed_obj_values, heuristic_solver, pho = 0.9):
            if k == 0:
                self.best_bound = self.__estimate_best_bound(heuristic_solver)

            self.relaxed_obj_values = relaxed_obj_values
            self.step_size = pho * self.sign * (self.best_bound - relaxed_obj_values) / (np.linalg.norm(self.subgrad))**2
            print("subgrad norm: ", (np.linalg.norm(self.subgrad))**2)

        def update_mulpier(self):
            self.mulpier += self.sign * self.step_size * self.subgrad
            for i in range(self.mulpier.size):
                if self.sense[i] == '<':
                    self.mulpier[0, i] = max(0, self.mulpier[0, i]) if self.sign == 1.0 else min(0, self.mulpier[0, i])
                elif self.sense[i] == '>':
                    self.mulpier[0, i] = min(0, self.mulpier[0, i]) if self.sign == 1.0 else max(0, self.mulpier[0, i])
            return self.mulpier

        def __estimate_best_bound(self, heuristic_solver):
            heuristic_solver.optimize()
            best_bound = heuristic_solver.get_objective_values()
            return best_bound

class SurrogateLagrangRelax(LagrangRelax):
    def __init__(self, model, relaxed_constrs, mulpier, r, big_m):
        super(SurrogateLagrangRelax, self).__init__(model, relaxed_constrs, mulpier)
        self.r, self.big_m = r, big_m
        assert(self.r > 0 and self.r < 1 and self.big_m >= 1)
    
    class DualityProb:
        def __init__(self, lagrang_relax):
            self.mulpier, self.sense = lagrang_relax.mulpier, lagrang_relax.sense
            self.sign = 1.0 if lagrang_relax.orig_model.getAttr("ModelSense") == 'Minimization' else -1.0
            self.r, self.big_m = lagrang_relax.r, lagrang_relax.big_m
        
        def get_subgrad(self, relaxed_expr):
            self.subgrad = np.array([relaxed_expr[i].getValue() for i in range(len(relaxed_expr))])
            return self.subgrad   
        
        def get_step_size(self, k, relaxed_obj_values, heuristic_solver):
            if k == 0:
                self.best_bound = self.__estimate_best_bound(heuristic_solver)
                self.relaxed_obj_values = relaxed_obj_values
                self.step_size = self.sign * (self.best_bound - self.relaxed_obj_values) / (np.linalg.norm(self.subgrad))**2
            
            else:
                self.p = 1 - 1 / ((k + 1)**self.r)
                self.alpha = 1 - 1 / (self.big_m * (k + 1)**self.p)
                self.step_size = self.alpha * self.last_step_size * np.linalg.norm(self.last_subgrad) / np.linalg.norm(self.subgrad)
            
            self.last_step_size, self.last_subgrad = self.step_size, self.subgrad.copy()
           
        def update_mulpier(self):
            self.mulpier += self.sign * self.step_size * self.subgrad
            for i in range(self.mulpier.size):
                if self.sense[i] == '<':
                    self.mulpier[0, i] = max(0, self.mulpier[0, i]) if self.sign == 1.0 else min(0, self.mulpier[0, i])
                elif self.sense[i] == '>':
                    self.mulpier[0, i] = min(0, self.mulpier[0, i]) if self.sign == 1.0 else max(0, self.mulpier[0, i])
            return self.mulpier

        def __estimate_best_bound(self, heuristic_solver):
            heuristic_solver.optimize()
            best_bound = heuristic_solver.get_objective_values()
            return best_bound
            
class LevLagrangRelax(LagrangRelax):
    def __init__(self, model, relaxed_constrs, mulpier, R):
        super(LevLagrangRelax, self).__init__(model, relaxed_constrs, mulpier)
        self.sigma, self.R = 0, R
        assert(self.R > 0 and self.sigma == 0)
    
    class DualityProb:
        def __init__(self, lagrang_relax):
            self.mulpier, self.sense = lagrang_relax.mulpier, lagrang_relax.sense
            self.sign = 1.0 if lagrang_relax.orig_model.getAttr("ModelSense") == 'Minimization' else -1.0
            self.sigma, self.R  = lagrang_relax.sigma, lagrang_relax.R
            self.rec_obj_values, self.rec_subgrad, self.rec_mulpier = [np.inf], [], []
            self.L, self.K = 0, [0]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
            
        def get_subgrad(self, relaxed_expr):
            self.subgrad = np.array([relaxed_expr[i].getValue() for i in range(len(relaxed_expr))])
            self.__get_delta(self.subgrad)
            return self.subgrad   
        
        def __get_delta(self, subgrad):
            self.delta = 0.5 * np.linalg.norm(subgrad) * self.R
            return self.delta
        
        def evaluate_objective(self, relaxed_obj_values, k): 
            self.relaxed_obj_values = relaxed_obj_values
            if self.sign * self.relaxed_obj_values > self.sign * self.rec_obj_values[k]:
                self.rec_obj_values = self.rec_obj_values + [self.relaxed_obj_values]
                self.rec_mulpier = self.rec_mulpier + [self.mulpier.copy()]
                self.rec_subgrad = self.rec_subgrad + [self.subgrad.copy()]
            else:
                self.rec_obj_values = self.rec_obj_values + [self.rec_obj_values[k]]
                self.rec_mulpier = self.rec_mulpier + [self.rec_mulpier[k-1].copy()]
                self.rec_subgrad = self.rec_subgrad + [self.rec_subgrad[k-1].copy()]
            
        def is_descent(self, k):
            if self.sign * self.relaxed_obj_values >= self.sign * (self.rec_obj_values[self.K[self.L]] - 0.5 * self.delta):
                self.K, self.sigma = self.K + [k], 0      
                self.L += 1  
            else:
                self.__is_oscillation(k)
        
        def __is_oscillation(self, k):
            if self.sigma > self.R:
                self.K, self.sigma, self.delta = self.K + [k], 0, 0.5 * self.delta
                self.mulpier, self.subgrad = self.rec_mulpier[k].copy(), self.rec_subgrad[k].copy()
                self.L += 1
        
        def get_step_size(self):     
            self.lev_obj_values = self.rec_obj_values[self.K[self.L] + 1] + self.sign * self.delta
            self.step_size = self.sign * (self.lev_obj_values + self.delta - self.relaxed_obj_values) / (np.linalg.norm(self.subgrad))**2
            print("rec obj values: ", self.rec_obj_values)
            print("relaxed obj values: ", self.relaxed_obj_values)
            print("L: ", self.L)
            print("K: ", self.K)   
            print("mulpier: ", self.mulpier)
            print("sub grad: ", self.subgrad)
            print("step size: ", self.step_size)
            print("delta ", self.delta)
            print(" ")

        def update_mulpier(self):
            self.z = self.mulpier + self.sign * self.step_size * self.subgrad
            for i in range(self.mulpier.size):
                if self.sense[i] == '<':
                    self.mulpier[0, i] = max(0, self.z[0, i]) if self.sign == 1.0 else min(0, self.z[0, i])
                elif self.sense[i] == '>':
                    self.mulpier[0, i] = min(0, self.z[0, i]) if self.sign == 1.0 else max(0, self.z[0, i])
            return self.mulpier

        def update_path(self):
            self.sigma = self.sigma + np.linalg.norm(self.z - self.mulpier, ord = 1)

    
        