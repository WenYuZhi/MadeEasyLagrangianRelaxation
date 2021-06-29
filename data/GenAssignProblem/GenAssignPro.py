import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

class GenAssignPro:
    def __init__(self, nAgent, nJob, A, b, costCoeff):
        self.nAgent, self.nJob, self.A, self.b, self.costCoeff = nAgent, nJob, A, b, costCoeff
        self.m = gp.Model('GAP')
        self.x = self.m.addVars(self.nAgent, self.nJob, name = 'x', vtype = GRB.BINARY)

    def addConstrsOneAgentCapacity(self):
        self.m.addConstrs(gp.quicksum(self.A[i,j] * self.x[i,j] for j in range(self.nJob)) <= self.b[i] for i in range(self.nAgent))   

        #self.m.addConstrs((gp.quicksum(self.w[f,s,self.feasiblePeriods[f,s,1]] for s in self.subNode[f][j]) <= 1 for f in self.flightsSet \
        #for j in self.flightsToSectorsSet[f] if self.destAirport[f] != j and len(self.subNode[f][j]) > 1), name='c6') 
    
    def addConstsOneJobToOneAgent(self):
        self.m.addConstrs(gp.quicksum(self.x[i,j] for i in range(self.nAgent)) == 1 for j in range(self.nJob)) 
    
    def addConstrs(self):
        self.addConstrsOneAgentCapacity()
        self.addConstsOneJobToOneAgent()

    def setObjective(self):
        obj = gp.LinExpr()
        for i in range(self.nAgent):
            obj += gp.quicksum(self.costCoeff[i,j] * self.x[i,j] for j in range(self.nJob))
        self.m.setObjective(obj, GRB.MINIMIZE)
    
    def solve(self):
        self.m.optimize()
        self.writeModel()
        self.getSoluation()
        self.objValues = self.m.getObjective().getValue()
        self.writeSoluation()
        return self.objValues
    
    def getSoluation(self):
        self.sol, self.xSol2D = self.m.getVars(), np.zeros((self.nAgent, self.nJob)) 
        for i in range(self.nAgent * self.nJob):
            indexAgent, indexJob = i // self.nJob, i % self.nJob
            self.xSol2D[indexAgent, indexJob] = self.sol[i].x
        return self.xSol2D

    def writeModel(self):
        self.m.write('GAP.lp')
    
    def writeSoluation(self):
        xsol2D = pd.DataFrame(self.xSol2D)
        xsol2D.to_csv('GAPSoluation.csv')

class GenAssignProLR(GenAssignPro):
    def addConstrs(self):
        self.addConstrsOneAgentCapacity()
        
    def setObjective(self, lamb):
        obj = gp.LinExpr()
        for i in range(self.nAgent):
            obj += gp.quicksum((self.costCoeff[i,j] + lamb[i]) * self.x[i,j] for j in range(self.nJob))
        obj -= lamb.sum()
        self.m.setObjective(obj, GRB.MINIMIZE)
    
    def computeSubgradients(self):
        self.subgradients = np.zeros(self.nJob)
        for i in range(self.nJob):
            self.subgradients[i] = self.xSol2D[:,i].sum() - 1
        return self.subgradients
    
    def getFeasibleByheuristic(self):
        xFeasibleSol = self.xSol2D.copy()
        return xFeasibleSol

class DualProb:
    def __init__(self, bestUpBound, lamd):
        self.bestUpBound, self.lamd = bestUpBound, lamd
        self.subgradientsNormIteration, self.stepSizeIteration, self.relaxedObjValuesIterations = [], [], []
        self.lamdIterations, self.subgradientsIteration, self.origObjValuesIterations = [], [], []
    
    def setSubgradients(self, subgradients):
        self.subgradients = subgradients
    
    def computeStepSize(self):
        self.stepSize = (self.bestUpBound - self.relaxedObjValues) / (np.linalg.norm(self.subgradients))**2

    def updateLamd(self):
        self.lamd += self.stepSize * self.subgradients
        return self.lamd

    def getRelaxedObjValues(self, origObjValues):
        self.origObjValues = origObjValues
        self.relaxedObjValues = origObjValues + (self.lamd * self.subgradients).sum()
        #print(origObjValues, (self.lamd * self.subgradients).sum())
    
    def moniterProcess(self):
        self.subgradientsNormIteration.append(np.linalg.norm(self.subgradients))
        self.stepSizeIteration.append(self.stepSize)
        self.relaxedObjValuesIterations.append(self.relaxedObjValues)
        self.lamdIterations.append(self.lamd)
        self.origObjValuesIterations.append(self.origObjValues)
        self.subgradientsIteration.append(self.subgradients)
    
    def printResults(self):
        print("subgradients norm: ", self.subgradientsNormIteration)
        print("step size: ", self.stepSizeIteration)
        print("relaxed ObjValues：", self.relaxedObjValuesIterations)
        print("orig ObjValues：", self.origObjValuesIterations)
        print("best up bound: ", self.bestUpBound)
        print("lamd: ", self.lamd)
        print("subgradients: ", self.subgradients)
    
    def printLog(self, fileName):
        with open(fileName, 'a') as f:
            for iterTime in range(len(self.subgradientsNormIteration)):
                f.write("iter times: " + str(iterTime) + '\n')
                f.write("step size: " + str(self.stepSizeIteration[iterTime]) + '  ')
                f.write("subgradients norm: " + str(self.subgradientsNormIteration[iterTime]) + '  ')
                f.write("best up bound: " + str(self.bestUpBound) + '  ')
                f.write("relaxed objective: " + str(self.relaxedObjValuesIterations[iterTime]) + '\n')
        temp = pd.DataFrame(self.lamdIterations).T
        temp.to_csv('lamd.csv')
        temp = pd.DataFrame(self.subgradientsIteration).T
        temp.to_csv('subgradients.csv')

class DualProbSurrogateLR(DualProb):
    def __init__(self, m, r, bestUpBound, lamd):
        self.m, self.r, self.bestUpBound, self.lamd = m, r, bestUpBound, lamd
        self.subgradientsNormIteration, self.stepSizeIteration, self.relaxedObjValuesIterations = [], [], []
        self.lamdIterations, self.origObjValuesIterations = [], []
    
    def computeStepSize(self, iterTimes):
        if iterTimes == 0:
            self.stepSize = (self.bestUpBound - self.relaxedObjValues) / (np.linalg.norm(self.subgradients))**2
        else:
            self.p = 1 - 1 / ((iterTimes + 1)**self.r)
            self.alpha = 1 - 1 / (self.m * (iterTimes + 1)**self.p)
            self.stepSize = self.alpha * self.stepSizeLast * np.linalg.norm(self.subgrdientsLast) / np.linalg.norm(self.subgradients)
    
    def setLast(self):
        self.subgrdientsLast, self.stepSizeLast = self.subgradients, self.stepSize